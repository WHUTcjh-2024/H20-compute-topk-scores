# H20 benchmark results

Measured 2026-10-24 on NVIDIA H20, commit `425c59e`.

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
| Git HEAD | 425c59eb913624e4746d4fd2ce5b4b0c1b49c9b7 |

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
| `_topk_log_softmax_kernel` | Online log-sum-exp in one vocab pass instead of separate max and sum-exp passes. |
| `_ranks_kernel` | `tl.range(..., num_stages=2)` to pipeline the vocab scan. |
| `_fill_logprob_token_ids_kernel` | Contiguous/aligned offset hints; measured ~18.4us and launch-bound. |

The fill kernel is launch-bound: its latency is ~18.4us for B=1..256 and
unchanged for 1/2/4 warps. No body change produced a measurable reduction;
the hint is retained and the kernel remains a non-bottleneck.

## Final performance

Reference: upstream vLLM 0.29.0. Candidate: this repository. Serial
measurement, CUDA events, warmup=20, repeats=200, median per case.

| Metric | Value |
| --- | ---: |
| Cases | 72 |
| ratio min | 1.006 |
| ratio p10 | 1.011 |
| ratio median | 1.114 |
| ratio mean | 1.105 |
| ratio p90 | 1.189 |
| ratio max | 1.202 |
| ratio < 0.8 | 0 |
| 0.8 <= ratio < 1.0 | 0 |
| ratio >= 1.0 | 72 |

All 72 cases are >=1.0. The V=32768 cases are 1.006-1.023; the larger
vocab cases are 1.09-1.202.

### Median ratio by B / V

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 1.011 | 1.175 | 1.188 |
| 8 | 1.019 | 1.178 | 1.196 |
| 64 | 1.013 | 1.120 | 1.123 |
| 256 | 1.012 | 1.095 | 1.106 |

| dtype | B | V | K | reference us | candidate us | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | 1 | 32768 | 1 | 112.98 | 110.86 | 1.019 |
| float16 | 1 | 32768 | 1 | 112.00 | 111.26 | 1.007 |
| bfloat16 | 1 | 32768 | 10 | 121.12 | 120.43 | 1.006 |
| float16 | 1 | 32768 | 10 | 120.46 | 119.79 | 1.006 |
| bfloat16 | 1 | 32768 | 50 | 123.90 | 121.97 | 1.016 |
| float16 | 1 | 32768 | 50 | 143.47 | 123.44 | 1.162 |
| bfloat16 | 1 | 131072 | 1 | 202.69 | 171.12 | 1.184 |
| float16 | 1 | 131072 | 1 | 202.91 | 171.36 | 1.184 |
| bfloat16 | 1 | 131072 | 10 | 211.10 | 179.86 | 1.174 |
| float16 | 1 | 131072 | 10 | 211.46 | 180.67 | 1.170 |
| bfloat16 | 1 | 131072 | 50 | 213.79 | 181.68 | 1.177 |
| float16 | 1 | 131072 | 50 | 214.53 | 183.25 | 1.171 |
| bfloat16 | 1 | 151936 | 1 | 220.83 | 184.40 | 1.198 |
| float16 | 1 | 151936 | 1 | 221.02 | 185.71 | 1.190 |
| bfloat16 | 1 | 151936 | 10 | 230.08 | 194.00 | 1.186 |
| float16 | 1 | 151936 | 10 | 229.63 | 194.54 | 1.180 |
| bfloat16 | 1 | 151936 | 50 | 232.48 | 195.46 | 1.189 |
| float16 | 1 | 151936 | 50 | 233.06 | 197.39 | 1.181 |
| bfloat16 | 8 | 32768 | 1 | 114.06 | 111.55 | 1.023 |
| float16 | 8 | 32768 | 1 | 114.18 | 112.27 | 1.017 |
| bfloat16 | 8 | 32768 | 10 | 123.36 | 121.12 | 1.018 |
| float16 | 8 | 32768 | 10 | 123.28 | 121.52 | 1.014 |
| bfloat16 | 8 | 32768 | 50 | 122.42 | 120.05 | 1.020 |
| float16 | 8 | 32768 | 50 | 122.29 | 120.00 | 1.019 |
| bfloat16 | 8 | 131072 | 1 | 206.10 | 172.88 | 1.192 |
| float16 | 8 | 131072 | 1 | 205.71 | 174.59 | 1.178 |
| bfloat16 | 8 | 131072 | 10 | 215.46 | 183.01 | 1.177 |
| float16 | 8 | 131072 | 10 | 214.70 | 183.28 | 1.171 |
| bfloat16 | 8 | 131072 | 50 | 215.07 | 181.66 | 1.184 |
| float16 | 8 | 131072 | 50 | 213.50 | 181.70 | 1.175 |
| bfloat16 | 8 | 151936 | 1 | 225.86 | 187.86 | 1.202 |
| float16 | 8 | 151936 | 1 | 224.62 | 187.54 | 1.198 |
| bfloat16 | 8 | 151936 | 10 | 234.22 | 196.99 | 1.189 |
| float16 | 8 | 151936 | 10 | 233.49 | 197.18 | 1.184 |
| bfloat16 | 8 | 151936 | 50 | 233.57 | 195.31 | 1.196 |
| float16 | 8 | 151936 | 50 | 233.28 | 195.07 | 1.196 |
| bfloat16 | 64 | 32768 | 1 | 117.01 | 115.46 | 1.013 |
| float16 | 64 | 32768 | 1 | 115.52 | 114.27 | 1.011 |
| bfloat16 | 64 | 32768 | 10 | 124.75 | 123.39 | 1.011 |
| float16 | 64 | 32768 | 10 | 123.76 | 122.11 | 1.013 |
| bfloat16 | 64 | 32768 | 50 | 126.29 | 123.90 | 1.019 |
| float16 | 64 | 32768 | 50 | 124.64 | 122.85 | 1.015 |
| bfloat16 | 64 | 131072 | 1 | 285.12 | 253.57 | 1.124 |
| float16 | 64 | 131072 | 1 | 278.66 | 248.74 | 1.120 |
| bfloat16 | 64 | 131072 | 10 | 291.20 | 259.92 | 1.120 |
| float16 | 64 | 131072 | 10 | 284.50 | 255.68 | 1.113 |
| bfloat16 | 64 | 131072 | 50 | 292.54 | 260.96 | 1.121 |
| float16 | 64 | 131072 | 50 | 286.56 | 256.99 | 1.115 |
| bfloat16 | 64 | 151936 | 1 | 323.94 | 287.36 | 1.127 |
| float16 | 64 | 151936 | 1 | 317.14 | 282.27 | 1.124 |
| bfloat16 | 64 | 151936 | 10 | 329.90 | 293.79 | 1.123 |
| float16 | 64 | 151936 | 10 | 324.03 | 289.42 | 1.120 |
| bfloat16 | 64 | 151936 | 50 | 330.27 | 293.15 | 1.127 |
| float16 | 64 | 151936 | 50 | 325.62 | 291.23 | 1.118 |
| bfloat16 | 256 | 32768 | 1 | 200.88 | 197.89 | 1.015 |
| float16 | 256 | 32768 | 1 | 195.47 | 193.18 | 1.012 |
| bfloat16 | 256 | 32768 | 10 | 207.70 | 204.77 | 1.014 |
| float16 | 256 | 32768 | 10 | 201.57 | 199.87 | 1.008 |
| bfloat16 | 256 | 32768 | 50 | 210.13 | 207.46 | 1.013 |
| float16 | 256 | 32768 | 50 | 204.86 | 203.25 | 1.008 |
| bfloat16 | 256 | 131072 | 1 | 727.09 | 663.89 | 1.095 |
| float16 | 256 | 131072 | 1 | 721.97 | 659.39 | 1.095 |
| bfloat16 | 256 | 131072 | 10 | 735.01 | 670.11 | 1.097 |
| float16 | 256 | 131072 | 10 | 729.82 | 667.52 | 1.093 |
| bfloat16 | 256 | 131072 | 50 | 738.66 | 673.98 | 1.096 |
| float16 | 256 | 131072 | 50 | 731.95 | 669.94 | 1.093 |
| bfloat16 | 256 | 151936 | 1 | 823.38 | 743.49 | 1.107 |
| float16 | 256 | 151936 | 1 | 816.66 | 738.74 | 1.105 |
| bfloat16 | 256 | 151936 | 10 | 833.42 | 753.36 | 1.106 |
| float16 | 256 | 151936 | 10 | 826.99 | 748.99 | 1.104 |
| bfloat16 | 256 | 151936 | 50 | 838.02 | 758.00 | 1.106 |
| float16 | 256 | 151936 | 50 | 828.14 | 751.06 | 1.103 |

