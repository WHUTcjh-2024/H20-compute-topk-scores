# H20 `compute_topk_scores` 复测报告（2026-10-26）

## 结论

当前复测代码提交 `d6c980c87c27a6bd62492194e42c6a6cfd923e1a` 已在 NVIDIA H20 上完成 correctness 与 performance 验证。

- CPU regression：1 passed
- GPU quick：60 passed
- GPU full：132 passed
- Performance：72 个 case
- SpeedUp min：0.982
- SpeedUp median：1.163
- SpeedUp mean：1.148
- SpeedUp max：1.297
- 65 / 72 个 case 不慢于 vLLM reference
- 7 / 72 个 case 略慢于 reference，均位于 `V=32768`，最差约慢 1.8%

`SpeedUp = reference latency / candidate latency`，数值越大表示 candidate 越快。

## 测试对象

| 项目 | 值 |
| --- | --- |
| Reference | vLLM 0.29.0 `vllm.v1.worker.gpu.sample.logprob.compute_topk_scores` |
| Candidate | `src/h20_topk/ops/compute_topk_scores.py` |
| Target GPU | NVIDIA H20 |
| 测试提交 | `d6c980c87c27a6bd62492194e42c6a6cfd923e1a` |
| 测试日期 | 2026-10-26 |

Candidate 保持 upstream host path，不重写 `compute_topk_scores`，不替换 `torch.topk`，不修改 upstream launch config。主要差异集中在三个 Triton kernel body：

```text
torch.topk
  -> _topk_log_softmax_kernel
  -> _ranks_kernel
  -> _fill_logprob_token_ids_kernel
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

环境检查中 `pip check` 与 `scripts/check_env.py --strict --require-gpu` 均通过。

## Correctness

### 测试结果

| 测试 | 结果 |
| --- | ---: |
| CPU `-inf` regression | 1 passed |
| GPU quick | 60 passed |
| GPU full | 132 passed |

主要覆盖：

- fp16 / bf16 / fp32
- 不同 B / V / K
- non-contiguous row stride
- ties
- custom token ids
- `logits_mode`
- empty batch
- `cu_num_logits`
- 首段全 `-inf`、后续存在有限值的 online-LSE 边界情况

其中 `-inf` regression 用例用于验证 online log-sum-exp 在前缀全为 `-inf` 时不会产生 NaN。

原始测试输出保存在：

```text
benchmark/retest-2026-10-26/tests-cpu.txt
benchmark/retest-2026-10-26/tests-quick.txt
benchmark/retest-2026-10-26/tests-full.txt
```

## Performance

### Workload

```text
dtype = {bf16, fp16}
B     = {1, 8, 64, 256}
V     = {32768, 131072, 151936}
K     = {1, 10, 50}

共 72 个 case
```

Reference 与 candidate 分别在独立进程中执行，使用相同固定随机种子。

测量方式：

```text
CUDA Event per-call timing
warmup = 20
repeats = 500
primary metric = median latency
```

### 汇总结果

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

7 个低于 1.0 的 case 全部位于 `V=32768`。对于更大的 vocab，candidate 的收益更明显。

原始 benchmark 数据保存在：

```text
benchmark/retest-2026-10-26/vllm.json
benchmark/retest-2026-10-26/candidate.json
benchmark/retest-2026-10-26/comparison.json
```

## 优化说明

### `_topk_log_softmax_kernel`

Upstream 对 vocab 做两遍扫描：先计算 row max，再计算 sum-exp。

Candidate 使用 online log-sum-exp，在一次扫描中维护 running max 和 running sum-exp，同时加入：

- full-block / tail split
- 完整 block 无 mask load
- `tl.range(num_stages=2, loop_unroll_factor=2)`
- all-masked prefix 的数值稳定性处理

这是主要性能收益来源。

### `_ranks_kernel`

主要调整：

- full-block / tail split
- 完整 block 无 mask load
- `tl.range(num_stages=4)`

### `_fill_logprob_token_ids_kernel`

该 kernel 的 body 调整较小。实验中其延迟接近 launch-bound，因此没有继续做复杂优化。

## 说明

本报告只描述当前 H20 复测结果。由于不同宿主机、驱动、Python 版本和代码提交之间的绝对时延不可直接比较，本报告不使用历史宿主机数据作为当前结果依据。

当前数据来自单次完整串行复测；对于接近 1.0 的小幅性能差异，不进一步解释为稳定回归或稳定收益。