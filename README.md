# compute_topk_scores — NVIDIA H20 算子优化

针对 vLLM 0.29.0 `compute_topk_scores` 路径的 NVIDIA H20 Triton kernel 优化。

- Reference：`vllm==0.29.0`，`vllm/v1/worker/gpu/sample/logprob.py`
- Candidate：`src/h20_topk/ops/compute_topk_scores.py`
- Target GPU：NVIDIA H20
  
### Correctness

| 测试 | 结果 |
| --- | ---: |
| CPU `-inf` regression | 1 passed |
| GPU quick | 60 passed |
| GPU full | 132 passed |

覆盖：

- fp16 / bf16 / float32
- 不同 B / V / K
- non-contiguous row stride
- ties
- custom token ids
- `logits_mode`
- empty batch
- `cu_num_logits`
- 首段全 `-inf`、后续存在有限值的 online-LSE 边界情况

### Performance

Benchmark active set：

```text
dtype = {bf16, fp16}
B     = {1, 8, 64, 256}
V     = {32768, 131072, 151936}
K     = {1, 10, 50}

共 72 个 case
```

测量方式：reference 和 candidate 分别独立进程执行；相同固定随机种子；CUDA Event per-call timing；warmup=20，repeats=500；报告 median latency。

`SpeedUp = reference latency / candidate latency`，数值越大越好。

| 指标 | 结果 |
| --- | ---: |
| SpeedUp min | **0.982** |
| SpeedUp p10 | 0.995 |
| SpeedUp median | **1.163** |
| SpeedUp mean | 1.148 |
| SpeedUp p90 | 1.272 |
| SpeedUp max | **1.297** |
| SpeedUp < 1.0 | 7 / 72 |
| SpeedUp >= 1.0 | **65 / 72** |

按 B / V 聚合后的 SpeedUp 中位数：

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 1.006 | 1.254 | 1.276 |
| 8 | 0.999 | 1.260 | 1.282 |
| 64 | 1.013 | 1.186 | 1.190 |
| 256 | 1.035 | 1.152 | 1.162 |

最差 case：

```text
fp16, B=8, V=32768, K=1
reference = 115.84 us
candidate = 117.94 us
SpeedUp   = 0.982
```

最佳 case：

```text
bf16, B=1, V=151936, K=1
SpeedUp = 1.297
```

7 个低于 1.0 的 case 全部位于 `V=32768`，最差约慢 1.8%；其余 65 个 case 不慢于 reference。

## 优化范围

保持 vLLM upstream host path，不重写 `compute_topk_scores`，不替换 `torch.topk`，不修改 upstream launch config。

Execution path：

```text
torch.topk
  -> _topk_log_softmax_kernel
  -> _ranks_kernel
  -> _fill_logprob_token_ids_kernel
```

host 侧仍保持 upstream 的：

```text
torch.topk
torch.cat
logits.gather
.to(torch.int32)
torch.zeros_like
masked_fill
```

Candidate 与 upstream 的主要差异集中在三个 Triton kernel body。

### `_topk_log_softmax_kernel`

Upstream 对整个 vocab 做两遍扫描：

```text
scan 1 -> row max
scan 2 -> sum(exp(logit - max))
```

Candidate 改为 online log-sum-exp：

```text
single scan
-> update running max
-> rescale running sum-exp
-> accumulate current block
```

同时加入：

- full-block / tail split，完整 block 不再执行 mask
- `tl.range(num_stages=2, loop_unroll_factor=2)`
- all-masked prefix 的 `-inf` 数值稳定性修复

这是主要性能收益来源。

### `_ranks_kernel`

优化内容：

- full-block / tail split
- 完整 block 无 mask load
- `tl.range(num_stages=4)`

### `_fill_logprob_token_ids_kernel`

只保留较小的 body 调整和连续/对齐 hint。

该 kernel 在实验中表现为 launch-bound / non-bottleneck，因此不继续投入复杂优化。

## Correctness 命令

```bash
python -m pytest -q tests/test_log_softmax_cpu.py

QUICK_MODE=1 python -m pytest -q \
  tests/test_compute_topk_scores.py -m compute_topk_scores

python -m pytest -q \
  tests/test_compute_topk_scores.py -m compute_topk_scores
```

## Benchmark 命令

Reference：

```bash
python scripts/bench_matrix.py --target vllm \
  --batches 1,8,64,256 \
  --vocabs 32768,131072,151936 \
  --ks 1,10,50 \
  --dtypes bfloat16,float16 \
  --warmup 20 \
  --repeats 500 \
  --output artifacts/retest/vllm.json
```

Candidate：

```bash
python scripts/bench_matrix.py --target candidate \
  --op src/h20_topk/ops/compute_topk_scores.py \
  --batches 1,8,64,256 \
  --vocabs 32768,131072,151936 \
  --ks 1,10,50 \
  --dtypes bfloat16,float16 \
  --warmup 20 \
  --repeats 500 \
  --output artifacts/retest/candidate.json
```

Compare：

```bash
python scripts/compare_bench.py \
  --reference artifacts/retest/vllm.json \
  --candidate artifacts/retest/candidate.json \
  --output artifacts/retest/comparison.json
```

## 复测环境

| 项目 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA H20 |
| Compute Capability | 9.0 |
| NVIDIA Driver | 580.65.06 |
| CUDA Driver | 13.0 |
| CUDA Toolkit | 13.0 |
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| vLLM | 0.29.0 |
| pytest | 9.1.1 |
