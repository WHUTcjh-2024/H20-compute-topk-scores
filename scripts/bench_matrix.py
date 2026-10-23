"""Frozen latency harness for vLLM ``compute_topk_scores`` and the optimized candidate.

One process measures one target. Inputs are generated from a fixed seed so both
targets see identical data. Device time comes from CUDA events; median latency
is the primary metric.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_ints(value):
    return [int(part) for part in value.split(",")]


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_fn(target, op_path=None):
    if target == "vllm":
        from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
        return compute_topk_scores

    if op_path is not None:
        spec = importlib.util.spec_from_file_location("candidate_op", op_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.compute_topk_scores

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from h20_topk import compute_topk_scores

    return compute_topk_scores


def summarize(values):
    values = sorted(values)
    n = len(values)
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": values[0],
        "p10": values[max(0, int(0.1 * n) - 1)],
        "p90": values[min(n - 1, int(0.9 * n))],
        "max": values[-1],
    }


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    device_ms = []
    wall_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        device_ms.append(start.elapsed_time(end))
    return result, summarize(device_ms), summarize(wall_ms)


def environment(target, op_path=None):
    import triton
    import vllm

    record = {
        "target": target,
        "candidate_op": str(op_path) if op_path is not None else None,
        "git_head": git_head(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "triton_path": triton.__file__,
        "python": sys.version,
        "measurement": "cuda_event_per_call_with_sync; median reported",
    }
    try:
        props = torch.cuda.get_device_properties(0)
        record["sm_count"] = props.multi_processor_count
        record["total_memory_mib"] = round(props.total_memory / 2**20)
    except Exception:
        pass
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("vllm", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--op",
        type=Path,
        default=None,
        help="load the candidate from this file instead of the h20_topk package",
    )
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,2,4,8,16,32,64,128,256"))
    parser.add_argument("--vocabs", type=parse_ints, default=parse_ints("32768,65536,131072,151936"))
    parser.add_argument("--ks", type=parse_ints, default=parse_ints("0,1,5,10,20,50"))
    parser.add_argument("--dtypes", default="bfloat16,float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    if args.op is not None and args.target != "candidate":
        parser.error("--op requires --target candidate")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")

    fn = load_fn(args.target, args.op)
    dtype_names = args.dtypes.split(",")
    cases = list(itertools.product(dtype_names, args.batches, args.vocabs, args.ks))
    if args.max_cases:
        cases = cases[: args.max_cases]

    source = Path(__file__).read_bytes()
    report = {
        "environment": environment(args.target, args.op),
        "harness_sha256": hashlib.sha256(source).hexdigest(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, (dtype_name, batch, vocab, k) in enumerate(cases, 1):
        dtype = DTYPES[dtype_name]
        torch.manual_seed(314159 + batch + vocab + k)
        logits = torch.randn(batch, vocab, dtype=dtype, device="cuda")
        sampled = torch.randint(vocab, (batch,), dtype=torch.int64, device="cuda")

        result, device, wall = measure(
            lambda: fn(logits, k, sampled), args.warmup, args.repeats
        )
        assert result.logprob_token_ids.shape == (batch, k + 1)
        assert result.logprobs.shape == (batch, k + 1)
        assert result.selected_token_ranks.shape == (batch,)

        record = {
            "dtype": dtype_name,
            "B": batch,
            "V": vocab,
            "K": k,
            "device_ms": device,
            "wall_ms": wall,
            "host_overhead_ms": wall["median"] - device["median"],
        }
        report["cases"].append(record)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(
            f"[{index}/{len(cases)}] {dtype_name} B={batch} V={vocab} K={k} "
            f"device={device['median']:.4f}ms wall={wall['median']:.4f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
