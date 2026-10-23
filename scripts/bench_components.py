"""Per-stage latency breakdown of upstream vLLM ``compute_topk_scores``."""

import argparse
import json
import statistics
from pathlib import Path

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_ints(value):
    return [int(part) for part in value.split(",")]


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=parse_ints, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    import vllm.v1.worker.gpu.sample.logprob as lp

    dtype = DTYPES[args.dtype]
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [
            (1, 131072),
            (8, 131072),
            (64, 131072),
            (256, 151936),
        ]
    )
    report = {
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }

    for batch, vocab in shapes:
        for k in (1, 10, 50):
            torch.manual_seed(1 + batch + vocab + k)
            logits = torch.randn(batch, vocab, dtype=dtype, device="cuda")
            sampled = torch.randint(vocab, (batch,), dtype=torch.int64, device="cuda")
            topk_ids = None

            def do_topk():
                nonlocal topk_ids
                topk_ids = torch.topk(logits, k, dim=-1).indices

            do_topk()
            ids = torch.cat((sampled.unsqueeze(-1), topk_ids), dim=1)
            ranks = torch.empty(batch, dtype=torch.int64, device="cuda")

            def do_ranks():
                lp._ranks_kernel[(batch,)](
                    ranks,
                    logits,
                    logits.stride(0),
                    sampled,
                    vocab,
                    BLOCK_SIZE=8192,
                )

            stages = {
                "topk_only": measure(do_topk, args.warmup, args.repeats),
                "cat_ids_only": measure(
                    lambda: torch.cat((sampled.unsqueeze(-1), topk_ids), dim=1),
                    args.warmup,
                    args.repeats,
                ),
                "log_softmax_gather": measure(
                    lambda: lp.compute_token_logprobs(logits, ids),
                    args.warmup,
                    args.repeats,
                ),
                "ranks_kernel": measure(do_ranks, args.warmup, args.repeats),
                "full_op": measure(
                    lambda: lp.compute_topk_scores(logits, k, sampled),
                    args.warmup,
                    args.repeats,
                ),
            }
            record = {"B": batch, "V": vocab, "K": k, "stages_ms": stages}
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
