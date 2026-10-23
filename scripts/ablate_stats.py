"""Ablation of the fused row-scan kernel: online/two-pass lse, rank fusion, splits."""

import torch
import triton
import triton.language as tl


@triton.jit
def stats_kernel(
    logits_ptr,
    stride,
    sampled_ptr,
    pmax_ptr,
    psum_ptr,
    prank_ptr,
    vocab_size,
    chunk,
    COMPUTE_LSE: tl.constexpr,
    COMPUTE_RANK: tl.constexpr,
    ONLINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    row_ptr = logits_ptr + row.to(tl.int64) * stride
    out_idx = row * tl.num_programs(1) + split

    token = tl.load(sampled_ptr + row).to(tl.int64)
    sval = tl.load(row_ptr + token).to(tl.float32)

    rmax = -float("inf")
    rsum = 0.0
    rank = 0

    if ONLINE:
        for off in tl.range(start, end, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < end
            x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
            if COMPUTE_LSE:
                nmax = tl.maximum(rmax, tl.max(x))
                scale = tl.where(rmax == -float("inf"), 0.0, tl.exp(rmax - nmax))
                rsum = rsum * scale + tl.sum(tl.where(mask, tl.exp(x - nmax), 0.0))
                rmax = nmax
            if COMPUTE_RANK:
                rank += tl.sum(((x >= sval) & mask).to(tl.int32))
    else:
        if COMPUTE_LSE:
            for off in tl.range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                rmax = tl.maximum(rmax, tl.max(x))
            for off in tl.range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                rsum += tl.sum(tl.where(mask, tl.exp(x - rmax), 0.0))
        if COMPUTE_RANK:
            for off in tl.range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                rank += tl.sum(((x >= sval) & mask).to(tl.int32))

    tl.store(pmax_ptr + out_idx, rmax)
    tl.store(psum_ptr + out_idx, rsum)
    tl.store(prank_ptr + out_idx, rank)


def bench(fn, w=20, r=100):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(r):
        a = torch.cuda.Event(True)
        b = torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def two_pass_rank_only(logits, sampled):
    """vLLM-style reference: separate lse (2 reads) + ranks (1 read)."""
    import vllm.v1.worker.gpu.sample.logprob as lp

    ids = sampled.unsqueeze(-1)
    t1 = bench(lambda: lp.compute_token_logprobs(logits, ids))
    rk = torch.empty(logits.shape[0], dtype=torch.int64, device="cuda")
    t2 = bench(
        lambda: lp._ranks_kernel[(logits.shape[0],)](
            rk, logits, logits.stride(0), sampled, logits.shape[1], BLOCK_SIZE=8192
        )
    )
    return t1 + t2


def main():
    for batch, vocab in ((256, 32768), (64, 131072), (8, 131072), (1, 131072)):
        logits = torch.randn(batch, vocab, dtype=torch.bfloat16, device="cuda")
        sampled = torch.randint(vocab, (batch,), dtype=torch.int64, device="cuda")
        print(f"\n=== B={batch} V={vocab} ===")
        print(
            f"  vLLM lse+ranks (3 reads, 2 launches): {two_pass_rank_only(logits, sampled):.4f} ms"
        )
        for splits in (1, 2, 4, 8, 16, 32):
            if vocab // splits < 4096:
                continue
            chunk = triton.cdiv(triton.cdiv(vocab, splits), 4096) * 4096
            pmax = torch.empty(batch, splits, dtype=torch.float32, device="cuda")
            psum = torch.empty(batch, splits, dtype=torch.float32, device="cuda")
            prank = torch.empty(batch, splits, dtype=torch.int32, device="cuda")
            for name, lse, rnk, online, block, warps in (
                ("fused-online", True, True, True, 4096, 8),
                ("lse-only", True, False, True, 4096, 8),
                ("rank-only", False, True, True, 4096, 8),
                ("fused-twopass", True, True, False, 4096, 8),
                ("fused-online-b2k-w4", True, True, True, 2048, 4),
                ("fused-online-b8k-w8", True, True, True, 8192, 8),
            ):
                fn = lambda: stats_kernel[(batch, splits)](
                    logits,
                    logits.stride(0),
                    sampled,
                    pmax,
                    psum,
                    prank,
                    vocab,
                    chunk,
                    COMPUTE_LSE=lse,
                    COMPUTE_RANK=rnk,
                    ONLINE=online,
                    BLOCK=block,
                    num_warps=warps,
                )
                t = bench(fn)
                print(
                    f"  S={splits:2d} {name:22s} BLOCK={block:5d} w={warps}: {t:.4f} ms"
                )


if __name__ == "__main__":
    main()
