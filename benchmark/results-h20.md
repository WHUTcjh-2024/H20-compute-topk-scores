# H20 benchmark results

Measured 2026-10-24 on NVIDIA H20, commit `42ef89b`.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA H20 |
| Driver | 580.105.08 |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| vLLM | 0.29.0 (reference) |
| Python | 3.12.3 |
| Git HEAD | 42ef89bdead05d9ee6e7cfb0fe635118aa20c878 |

## Correctness

- Quick suite: 57 passed
- Full suite: 129 passed

## Isolated upstream kernel baseline

Median latency in microseconds. Kernel-only measurements use preallocated
output buffers; `torch.topk` and the full upstream op allocate their own outputs.

| B | V | K | dtype | _topk_log_softmax | _ranks | _fill | torch.topk | full op |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 131072 | 10 | bfloat16 | 106.80 | 31.46 | 18.53 | 58.11 | 214.10 |
| 8 | 131072 | 10 | bfloat16 | 108.10 | 31.82 | 18.38 | 70.94 | 215.34 |
| 64 | 131072 | 10 | bfloat16 | 108.94 | 32.24 | 18.21 | 161.26 | 292.19 |
| 256 | 151936 | 20 | bfloat16 | 208.91 | 67.41 | 18.45 | 574.35 | 834.88 |

## Optimization

All upstream launch configurations and the host path are preserved. Only the
three target kernel bodies are changed.

| Kernel | Body change |
| --- | --- |
| `_topk_log_softmax_kernel` | Online log-sum-exp in one vocab pass, full-block/tail split, `tl.range(..., num_stages=2, loop_unroll_factor=2)`. |
| `_ranks_kernel` | Full-block/tail split, `tl.range(..., num_stages=4)` for the vocab scan. |
| `_fill_logprob_token_ids_kernel` | Loads-first ordering and contiguous/aligned hints; measured ~18.7us and launch-bound. |

The fill kernel is launch-bound: loads-first, write-through and no-custom
variants all measured 18.6-18.8us, while an empty kernel is ~13.1us. The
loads-first ordering is retained as the minimal body change; the kernel
remains a non-bottleneck.

## Final performance

Reference: upstream vLLM 0.29.0. Candidate: this repository. Serial
measurement, CUDA events, warmup=20, repeats=500, median per case.

| Metric | Value |
| --- | ---: |
| Cases | 72 |
| ratio min | 0.980 |
| ratio p10 | 0.990 |
| ratio median | 1.172 |
| ratio mean | 1.149 |
| ratio p90 | 1.262 |
| ratio max | 1.280 |
| ratio < 0.8 | 0 |
| 0.8 <= ratio < 1.0 | 10 |
| ratio >= 1.0 | 62 |

All cases are >=0.98. The 10 cases below 1.0 are V=32768, B=1/8, in
0.980-1.001; the larger-vocab cases are 1.09-1.28.

### Median ratio by B / V

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 0.986 | 1.236 | 1.262 |
| 8 | 0.992 | 1.251 | 1.273 |
| 64 | 1.043 | 1.192 | 1.197 |
| 256 | 1.045 | 1.158 | 1.169 |

| dtype | B | V | K | reference us | candidate us | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | 1 | 32768 | 1 | 109.68 | 111.07 | 0.987 |
| float16 | 1 | 32768 | 1 | 109.82 | 109.73 | 1.001 |
| bfloat16 | 1 | 32768 | 10 | 117.95 | 119.74 | 0.985 |
| float16 | 1 | 32768 | 10 | 118.13 | 120.48 | 0.980 |
| bfloat16 | 1 | 32768 | 50 | 120.48 | 122.56 | 0.983 |
| float16 | 1 | 32768 | 50 | 120.62 | 122.24 | 0.987 |
| bfloat16 | 1 | 131072 | 1 | 199.46 | 159.30 | 1.252 |
| float16 | 1 | 131072 | 1 | 199.17 | 159.39 | 1.250 |
| bfloat16 | 1 | 131072 | 10 | 208.00 | 168.35 | 1.236 |
| float16 | 1 | 131072 | 10 | 208.78 | 168.80 | 1.237 |
| bfloat16 | 1 | 131072 | 50 | 209.54 | 170.48 | 1.229 |
| float16 | 1 | 131072 | 50 | 210.98 | 170.98 | 1.234 |
| bfloat16 | 1 | 151936 | 1 | 218.11 | 170.56 | 1.279 |
| float16 | 1 | 151936 | 1 | 218.34 | 171.65 | 1.272 |
| bfloat16 | 1 | 151936 | 10 | 227.09 | 179.94 | 1.262 |
| float16 | 1 | 151936 | 10 | 226.66 | 197.71 | 1.146 |
| bfloat16 | 1 | 151936 | 50 | 229.18 | 181.70 | 1.261 |
| float16 | 1 | 151936 | 50 | 230.30 | 183.34 | 1.256 |
| bfloat16 | 8 | 32768 | 1 | 111.23 | 111.07 | 1.001 |
| float16 | 8 | 32768 | 1 | 112.06 | 113.18 | 0.990 |
| bfloat16 | 8 | 32768 | 10 | 120.22 | 120.30 | 0.999 |
| float16 | 8 | 32768 | 10 | 120.13 | 121.50 | 0.989 |
| bfloat16 | 8 | 32768 | 50 | 119.36 | 120.58 | 0.990 |
| float16 | 8 | 32768 | 50 | 119.57 | 120.26 | 0.994 |
| bfloat16 | 8 | 131072 | 1 | 202.78 | 161.12 | 1.259 |
| float16 | 8 | 131072 | 1 | 202.43 | 161.12 | 1.256 |
| bfloat16 | 8 | 131072 | 10 | 212.35 | 170.94 | 1.242 |
| float16 | 8 | 131072 | 10 | 211.71 | 170.18 | 1.244 |
| bfloat16 | 8 | 131072 | 50 | 211.76 | 168.77 | 1.255 |
| float16 | 8 | 131072 | 50 | 210.05 | 168.45 | 1.247 |
| bfloat16 | 8 | 151936 | 1 | 223.17 | 174.29 | 1.280 |
| float16 | 8 | 151936 | 1 | 221.14 | 173.23 | 1.277 |
| bfloat16 | 8 | 151936 | 10 | 231.97 | 182.98 | 1.268 |
| float16 | 8 | 151936 | 10 | 230.78 | 181.25 | 1.273 |
| bfloat16 | 8 | 151936 | 50 | 230.82 | 181.28 | 1.273 |
| float16 | 8 | 151936 | 50 | 230.02 | 180.80 | 1.272 |
| bfloat16 | 64 | 32768 | 1 | 117.79 | 112.03 | 1.051 |
| float16 | 64 | 32768 | 1 | 117.04 | 112.32 | 1.042 |
| bfloat16 | 64 | 32768 | 10 | 124.85 | 123.41 | 1.012 |
| float16 | 64 | 32768 | 10 | 123.23 | 121.70 | 1.013 |
| bfloat16 | 64 | 32768 | 50 | 126.82 | 121.44 | 1.044 |
| float16 | 64 | 32768 | 50 | 125.41 | 120.02 | 1.045 |
| bfloat16 | 64 | 131072 | 1 | 285.18 | 238.75 | 1.194 |
| float16 | 64 | 131072 | 1 | 279.63 | 233.26 | 1.199 |
| bfloat16 | 64 | 131072 | 10 | 291.78 | 245.02 | 1.191 |
| float16 | 64 | 131072 | 10 | 286.53 | 240.29 | 1.192 |
| bfloat16 | 64 | 131072 | 50 | 293.28 | 246.72 | 1.189 |
| float16 | 64 | 131072 | 50 | 288.35 | 242.21 | 1.191 |
| bfloat16 | 64 | 151936 | 1 | 324.70 | 270.72 | 1.199 |
| float16 | 64 | 151936 | 1 | 318.43 | 265.18 | 1.201 |
| bfloat16 | 64 | 151936 | 10 | 330.88 | 276.99 | 1.195 |
| float16 | 64 | 151936 | 10 | 325.44 | 271.76 | 1.198 |
| bfloat16 | 64 | 151936 | 50 | 330.91 | 276.67 | 1.196 |
| float16 | 64 | 151936 | 50 | 327.38 | 274.02 | 1.195 |
| bfloat16 | 256 | 32768 | 1 | 200.99 | 192.80 | 1.042 |
| float16 | 256 | 32768 | 1 | 196.54 | 187.87 | 1.046 |
| bfloat16 | 256 | 32768 | 10 | 208.48 | 199.78 | 1.044 |
| float16 | 256 | 32768 | 10 | 203.62 | 194.59 | 1.046 |
| bfloat16 | 256 | 32768 | 50 | 210.82 | 201.98 | 1.044 |
| float16 | 256 | 32768 | 50 | 206.86 | 197.71 | 1.046 |
| bfloat16 | 256 | 131072 | 1 | 729.86 | 628.90 | 1.161 |
| float16 | 256 | 131072 | 1 | 723.62 | 625.02 | 1.158 |
| bfloat16 | 256 | 131072 | 10 | 736.24 | 636.99 | 1.156 |
| float16 | 256 | 131072 | 10 | 733.26 | 632.78 | 1.159 |
| bfloat16 | 256 | 131072 | 50 | 740.03 | 641.18 | 1.154 |
| float16 | 256 | 131072 | 50 | 735.52 | 634.78 | 1.159 |
| bfloat16 | 256 | 151936 | 1 | 824.96 | 704.45 | 1.171 |
| float16 | 256 | 151936 | 1 | 820.77 | 699.62 | 1.173 |
| bfloat16 | 256 | 151936 | 10 | 834.50 | 715.76 | 1.166 |
| float16 | 256 | 151936 | 10 | 829.39 | 709.98 | 1.168 |
| bfloat16 | 256 | 151936 | 50 | 839.86 | 720.54 | 1.166 |
| float16 | 256 | 151936 | 50 | 832.26 | 711.14 | 1.170 |

