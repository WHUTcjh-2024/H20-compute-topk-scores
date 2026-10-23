#!/usr/bin/env python3
"""Report the H20 experiment environment without requiring a GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = ("torch", "triton", "vllm")
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def run(cmd, timeout=20):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def os_release():
    text = read_text("/etc/os-release")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def git_info():
    commit = run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    branch = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def driver_from_proc():
    text = read_text("/proc/driver/nvidia/version")
    if not text:
        return None
    match = re.search(r"for x86_64\s+([0-9][0-9.]*)", text)
    return match.group(1) if match else None


def gpu_model_from_proc():
    base = Path("/proc/driver/nvidia/gpus")
    if not base.is_dir():
        return None
    for path in sorted(base.glob("*/information")):
        text = read_text(path)
        match = re.search(r"^Model:\s*(.+)$", text or "", re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def nvidia_smi_info():
    path = shutil.which("nvidia-smi")
    info = {"path": path, "usable": False, "driver": None, "cuda_version": None}
    fallback = Path("/usr/bin/nvidia-smi")
    if fallback.exists():
        info["file"] = (
            f"{fallback} ({fallback.stat().st_size} bytes, "
            f"executable={os.access(fallback, os.X_OK)})"
        )
    if not path:
        return info
    output = run(["nvidia-smi"], timeout=15)
    if output:
        info["usable"] = True
        match = re.search(r"Driver Version:\s*([0-9.]+)", output)
        if match:
            info["driver"] = match.group(1)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
        if match:
            info["cuda_version"] = match.group(1)
    return info


def cuda_toolkits():
    root = Path("/usr/local")
    found = []
    seen = set()
    if not root.is_dir():
        return found
    for path in sorted(root.glob("cuda*")):
        nvcc = path / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        version = None
        output = run([str(nvcc), "--version"])
        match = re.search(r"release\s+([0-9.]+)", output or "")
        if match:
            version = match.group(1)
        found.append({"path": str(path), "real_path": str(real), "nvcc": version})
    return found


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def import_status(name):
    try:
        module = __import__(name)
    except Exception as exc:  # report, never crash
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "dist_version": package_version(name),
        }
    return {
        "ok": True,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
        "dist_version": package_version(name),
    }


def collect():
    smi = nvidia_smi_info()
    toolkits = cuda_toolkits()
    packages = {
        name: import_status(name)
        for name in ("torch", "triton", "vllm", "pytest", "h20_topk")
    }

    cuda_available = False
    device_count = 0
    device_name = None
    compute_capability = None
    if packages["torch"]["ok"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = torch.cuda.device_count()
        packages["torch"]["cuda_version"] = torch.version.cuda
        packages["torch"]["cuda_available"] = cuda_available
        packages["torch"]["device_count"] = device_count
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            compute_capability = f"{capability[0]}.{capability[1]}"
            packages["torch"]["device_name"] = device_name

    return {
        "hostname": socket.gethostname(),
        "os": os_release() or platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "git": git_info(),
        "nvidia_smi": smi,
        "nvidia_driver": smi["driver"] or driver_from_proc(),
        "cuda_driver_version": smi["cuda_version"],
        "gpu_model": device_name or gpu_model_from_proc(),
        "compute_capability": compute_capability,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "cuda_toolkits": toolkits,
        "cuda_toolkit_pip": package_version("cuda-toolkit"),
        "packages": packages,
    }


def field(label, value):
    if value is None or value == "":
        value = "unknown"
    print(f"{label:<28}: {value}")


def render(info):
    torch = info["packages"]["torch"]
    print("=" * 68)
    print("H20 experiment environment check")
    print("=" * 68)
    print("[system]")
    field("hostname", info["hostname"])
    field("os", info["os"])
    field("kernel", info["kernel"])
    field("python", info["python"])
    field("python executable", info["python_executable"])
    field("python prefix", info["python_prefix"])
    field("git branch", info["git"]["branch"])
    field("git commit", info["git"]["commit"])
    field("git dirty files", info["git"]["dirty_files"])

    print("[cuda]")
    field("cuda available", info["cuda_available"])
    field("gpu model", info["gpu_model"])
    field("compute capability", info["compute_capability"])
    field("nvidia driver", info["nvidia_driver"])
    field(
        "cuda driver version",
        info["cuda_driver_version"] or "unknown (nvidia-smi unavailable)",
    )
    field("torch.version.cuda", torch.get("cuda_version"))
    field("pip cuda-toolkit", info["cuda_toolkit_pip"])
    if info["cuda_toolkits"]:
        for toolkit in info["cuda_toolkits"]:
            field("cuda toolkit", f"{toolkit['path']} (nvcc {toolkit['nvcc']})")
    else:
        field("cuda toolkit", None)
    field("nvidia-smi", info["nvidia_smi"]["path"] or "not usable")
    if "file" in info["nvidia_smi"]:
        field("nvidia-smi file", info["nvidia_smi"]["file"])

    print("[packages]")
    for name in ("torch", "triton", "vllm", "pytest", "h20_topk"):
        package = info["packages"][name]
        if package["ok"]:
            version = package.get("version") or package.get("dist_version")
            field(name, f"{version} ({package.get('path')})")
        else:
            detail = package["error"].splitlines()[0][:160]
            field(name, f"FAILED: {detail}")

    if not info["cuda_available"]:
        print()
        print("[note] CUDA is unavailable in this session. This is expected in")
        print("       CPU-only AutoDL mode; the versions and imports above are")
        print("       still valid. GPU stages are skipped by scripts/smoke_test.sh.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if torch, triton or vllm cannot be imported",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit 3 when CUDA is unavailable (default: still exit 0)",
    )
    args = parser.parse_args()

    info = collect()
    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        render(info)

    if args.strict:
        missing = [name for name in CORE_PACKAGES if not info["packages"][name]["ok"]]
        if missing:
            print(
                f"[check_env] missing core packages: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
    if args.require_gpu and not info["cuda_available"]:
        print("[check_env] no CUDA device available", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
