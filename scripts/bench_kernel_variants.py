"""Serial launch-config screening for the three upstream vLLM 0.29.0 kernels.

One process, one variant per measurement. This is a screening pass; final
numbers should be re-measured with the chosen config in a single sequential run.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton

CASES = [
    (1, 131072, 10),
    (8, 131072, 10),
    (64, 131072, 10),
    (256, 151936, 20),
]
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

CONFIGS = {
    "topk_log_softmax": [
        (1024, 4),
        (2048, 4),
        (4096, 4),
        (4096, 8),
        (8192, 8),
        (8192, 16),
    ],
    "ranks": [
        (8192, 4),
        (16384, 4),
        (16384, 8),
        (32768, 8),
        (32768, 16),
    ],
    "fill": [
        (1, 1),
        (1, 2),
        (1, 4),
    ],
}


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def summarize_us(values_ms):
    values = sorted(values_ms)
    n = len(values)
    return {
        "median_us": statistics.median(values) * 1000.0,
        "mean_us": statistics.fmean(values) * 1000.0,
        "min_us": values[0] * 1000.0,
        "p10_us": values[max(0, int(0.1 * n) - 1)] * 1000.0,
        "p90_us": values[min(n - 1, int(0.9 * n))] * 1000.0,
        "max_us": values[-1] * 1000.0,
    }


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values_ms.append(start.elapsed_time(end))
    return summarize_us(values_ms)


def make_inputs(batch, vocab, topk):
    torch.manual_seed(314159 + batch + vocab + topk)
    logits = torch.randn(batch, vocab, dtype=DTYPE, device="cuda")
    sampled = torch.randint(vocab, (batch,), dtype=torch.int64, device="cuda")
    if topk > 0:
        topk_ids = torch.topk(logits, topk, dim=-1).indices
    else:
        topk_ids = torch.empty((batch, 0), dtype=torch.int64, device="cuda")
    return logits, sampled, topk_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    import vllm
    import vllm.v1.worker.gpu.sample.logprob as lp

    report = {
        "reference": "vllm==0.29.0",
        "dtype": DTYPE_NAME,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "git_head": git_head(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "configs": [],
    }

    for kernel, configs in CONFIGS.items():
        for block_size, num_warps in configs:
            for batch, vocab, topk in CASES:
                logits, sampled, topk_ids = make_inputs(batch, vocab, topk)
                topk_block = min(triton.next_power_of_2(max(1, topk)), 1024)
                padded_cols = max(1, triton.next_power_of_2(topk))

                if kernel == "topk_log_softmax":
                    out = torch.empty(
                        (batch, topk), dtype=torch.float32, device="cuda"
                    )

                    def fn():
                        lp._topk_log_softmax_kernel[(batch,)](
                            out,
                            logits,
                            logits.stride(0),
                            topk_ids,
                            topk,
                            vocab,
                            BLOCK_SIZE=block_size,
                            TOPK_BLOCK_SIZE=topk_block,
                            num_warps=num_warps,
                        )

                elif kernel == "ranks":
                    out = torch.empty(batch, dtype=torch.int64, device="cuda")

                    def fn():
                        lp._ranks_kernel[(batch,)](
                            out,
                            logits,
                            logits.stride(0),
                            sampled,
                            vocab,
                            BLOCK_SIZE=block_size,
                            num_warps=num_warps,
                        )

                else:
                    topk_i32 = topk_ids.to(torch.int32)
                    out = torch.empty(
                        (batch, 1 + topk), dtype=torch.int64, device="cuda"
                    )
                    mask = torch.empty(
                        (batch, 1 + topk), dtype=torch.bool, device="cuda"
                    )
                    mapping = torch.arange(
                        batch, dtype=torch.int32, device="cuda"
                    )
                    num_per_req = torch.zeros(
                        batch, dtype=torch.int32, device="cuda"
                    )
                    per_req = torch.empty(
                        (batch, 1), dtype=torch.int32, device="cuda"
                    )

                    def fn():
                        lp._fill_logprob_token_ids_kernel[(batch,)](
                            out,
                            out.stride(0),
                            mask,
                            mask.stride(0),
                            sampled,
                            topk_i32,
                            topk_i32.stride(0),
                            mapping,
                            num_per_req,
                            per_req,
                            per_req.stride(0),
                            NUM_TOPK=topk,
                            PADDED_COLS=padded_cols,
                            num_warps=num_warps,
                        )

                try:
                    stats = measure(fn, args.warmup, args.repeats)
                except Exception as exc:
                    record = {
                        "kernel": kernel,
                        "block_size": block_size,
                        "num_warps": num_warps,
                        "B": batch,
                        "V": vocab,
                        "K": topk,
                        "dtype": DTYPE_NAME,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["configs"].append(record)
                    print(
                        f"{kernel:18s} block={block_size:6d} warps={num_warps:2d} "
                        f"B={batch:3d} V={vocab:6d} K={topk:2d} ERROR {exc}",
                        flush=True,
                    )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, indent=2), encoding="utf-8"
                    )
                    continue

                record = {
                    "kernel": kernel,
                    "block_size": block_size,
                    "num_warps": num_warps,
                    "B": batch,
                    "V": vocab,
                    "K": topk,
                    "dtype": DTYPE_NAME,
                    "stats": stats,
                }
                report["configs"].append(record)
                print(
                    f"{kernel:18s} block={block_size:6d} warps={num_warps:2d} "
                    f"B={batch:3d} V={vocab:6d} K={topk:2d} "
                    f"median={stats['median_us']:.2f}us",
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )


if __name__ == "__main__":
    main()
