"""Isolated latency baseline for the three upstream vLLM 0.29.0 kernels.

Reference: ``vllm.v1.worker.gpu.sample.logprob``.
No candidate implementation is imported here.

Kernel-only measurements use preallocated output buffers, so allocation is not
part of the reported latency. ``torch.topk`` and the full upstream
``compute_topk_scores`` allocate their own outputs and are measured as full
operations.
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
        "cases": [],
    }

    for batch, vocab, topk in CASES:
        torch.manual_seed(314159 + batch + vocab + topk)
        logits = torch.randn(batch, vocab, dtype=DTYPE, device="cuda")
        sampled = torch.randint(vocab, (batch,), dtype=torch.int64, device="cuda")
        if topk > 0:
            topk_ids = torch.topk(logits, topk, dim=-1).indices
        else:
            topk_ids = torch.empty((batch, 0), dtype=torch.int64, device="cuda")
        topk_ids_i32 = topk_ids.to(torch.int32)

        logsoftmax_out = torch.empty((batch, topk), dtype=torch.float32, device="cuda")
        ranks_out = torch.empty(batch, dtype=torch.int64, device="cuda")
        fill_out = torch.empty((batch, 1 + topk), dtype=torch.int64, device="cuda")
        fill_mask = torch.empty((batch, 1 + topk), dtype=torch.bool, device="cuda")
        expanded_idx_mapping = torch.arange(batch, dtype=torch.int32, device="cuda")
        num_per_req = torch.zeros(batch, dtype=torch.int32, device="cuda")
        per_req = torch.empty((batch, 1), dtype=torch.int32, device="cuda")

        topk_block = min(triton.next_power_of_2(max(1, topk)), 1024)
        padded_cols = max(1, triton.next_power_of_2(topk))

        def topk_log_softmax():
            lp._topk_log_softmax_kernel[(batch,)](
                logsoftmax_out,
                logits,
                logits.stride(0),
                topk_ids,
                topk,
                vocab,
                BLOCK_SIZE=1024,
                TOPK_BLOCK_SIZE=topk_block,
            )

        def ranks():
            lp._ranks_kernel[(batch,)](
                ranks_out,
                logits,
                logits.stride(0),
                sampled,
                vocab,
                BLOCK_SIZE=8192,
            )

        def fill_logprob_token_ids():
            lp._fill_logprob_token_ids_kernel[(batch,)](
                fill_out,
                fill_out.stride(0),
                fill_mask,
                fill_mask.stride(0),
                sampled,
                topk_ids_i32,
                topk_ids_i32.stride(0),
                expanded_idx_mapping,
                num_per_req,
                per_req,
                per_req.stride(0),
                NUM_TOPK=topk,
                PADDED_COLS=padded_cols,
            )

        stages = {
            "topk_log_softmax": measure(
                topk_log_softmax, args.warmup, args.repeats
            ),
            "ranks": measure(ranks, args.warmup, args.repeats),
            "fill_logprob_token_ids": measure(
                fill_logprob_token_ids, args.warmup, args.repeats
            ),
            "torch_topk": measure(
                lambda: torch.topk(logits, topk, dim=-1).indices,
                args.warmup,
                args.repeats,
            ),
            "full_compute_topk_scores": measure(
                lambda: lp.compute_topk_scores(logits, topk, sampled),
                args.warmup,
                args.repeats,
            ),
        }
        record = {
            "B": batch,
            "V": vocab,
            "K": topk,
            "dtype": DTYPE_NAME,
            "stages": stages,
        }
        report["cases"].append(record)
        print(
            f"B={batch} V={vocab} K={topk} {DTYPE_NAME} "
            f"topk_log_softmax={stages['topk_log_softmax']['median_us']:.2f}us "
            f"ranks={stages['ranks']['median_us']:.2f}us "
            f"fill={stages['fill_logprob_token_ids']['median_us']:.2f}us "
            f"topk={stages['torch_topk']['median_us']:.2f}us "
            f"full={stages['full_compute_topk_scores']['median_us']:.2f}us",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
