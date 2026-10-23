# compute_topk_scores — NVIDIA H20 优化实验

本仓库用于研究和验证 vLLM `compute_topk_scores` 算子在以下环境中的优化效果：

- GPU：NVIDIA H20
- CUDA Toolkit：13.0
- NVIDIA Driver：595.71.05
- Baseline：vLLM 0.29.0 `vllm/v1/worker/gpu/sample/logprob.py`

## 优化对象

当前主要关注 vLLM `compute_topk_scores` 路径中的三个 Triton kernel：

- `_topk_log_softmax_kernel`
- `_ranks_kernel`
- `_fill_logprob_token_ids_kernel`

原始执行路径大致为：

```text
torch.topk
  -> _topk_log_softmax_kernel   # row max + sum-exp + gather
  -> _ranks_kernel              # sampled token rank
  -> _fill_logprob_token_ids_kernel（custom token id path）
```

当前实现保留 `torch.topk`，重点优化 logits 扫描、log-softmax 统计、rank 计算以及相关 kernel launch / memory access 开销。

## 当前优化思路

vLLM 原实现中，log-softmax 与 rank 计算会重复扫描 logits。

当前实验主要尝试：

- 融合 log-softmax 统计与 rank 计算；
- 减少 logits 的重复全量读取；
- 降低 kernel launch 数量；
- 在小 batch / 大 vocab 场景下通过 split 提高 GPU 并行度；
- 保持与 vLLM `compute_topk_scores` 输出语义一致。

## Benchmark 方法

性能比较以 vLLM 原实现作为 baseline，在相同输入、dtype、shape、warmup 和重复次数下进行测量。

主要流程：

```text
vLLM baseline
    -> correctness check
    -> candidate implementation
    -> same workload benchmark
    -> latency comparison
```

性能比定义为：

```text
ratio = vLLM latency / optimized latency
```

同时记录不同 batch size、vocab size、top-k 和 dtype 下的性能变化，避免只依赖单一 shape 的结果。

## 目录

```text
src/flaggems_vllm/ops/compute_topk_scores.py  # 优化实现
tests/test_compute_topk_scores.py             # correctness tests
benchmark/test_compute_topk_scores.py         # vLLM 对比 benchmark
scripts/bench_components.py                   # vLLM 各阶段耗时拆解
scripts/ablate_stats.py                       # scan/rank fusion 与 split 消融
scripts/bench_matrix.py                       # 多 workload 延迟矩阵
```
