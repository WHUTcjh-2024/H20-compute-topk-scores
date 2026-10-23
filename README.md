# compute_topk_scores — NVIDIA H20 算子优化工程

本仓库独立研究和优化 vLLM 0.29.0 `compute_topk_scores` 路径中的三个 Triton kernel：

- `_topk_log_softmax_kernel`
- `_ranks_kernel`
- `_fill_logprob_token_ids_kernel`

## 术语与范围

- **reference**：upstream `vllm==0.29.0` 的 `vllm/v1/worker/gpu/sample/logprob.py`。
- **candidate**：本仓库 `src/h20_topk` 中的实现。
- 本仓库不依赖 FlagGems / FlagGems-vLLM；未来与 FlagGems-vLLM 的集成 **out of scope**。
- 当前 fused 实现仅作为 candidate / experiment，尚未证明是最终有效方案。新 H20 上会先独立测量三个 upstream kernel 的 baseline，再决定是否采用 fusion。
- `artifacts/` 中的历史数据只用于记录，不作为新 H20 的最终性能结果。

## 优化对象

原始执行路径大致为：

```text
torch.topk
  -> _topk_log_softmax_kernel   # row max + sum-exp + gather
  -> _ranks_kernel              # sampled token rank
  -> _fill_logprob_token_ids_kernel（custom token id path）
```

当前 candidate 保留 `torch.topk`，重点优化 logits 扫描、log-softmax 统计、rank 计算以及相关 kernel launch / memory access 开销。

## 当前 candidate 思路

vLLM 原实现中，log-softmax 与 rank 计算会重复扫描 logits。

当前 experiment 主要尝试：

- 融合 log-softmax 统计与 rank 计算；
- 减少 logits 的重复全量读取；
- 降低 kernel launch 数量；
- 在小 batch / 大 vocab 场景下通过 split 提高 GPU 并行度；
- 保持与 vLLM `compute_topk_scores` 输出语义一致。

这些只是待验证的 candidate 设计，不代表已经取得收益；是否保留 fusion 要等新 H20 上的独立 baseline 和对比结果。

## Benchmark 方法

reference 固定为 upstream vLLM 0.29.0，candidate 为本仓库实现。在相同输入、dtype、shape、warmup 和重复次数下测量：

```text
reference (vLLM 0.29.0)
    -> correctness check
    -> candidate (this repository)
    -> same workload benchmark
    -> latency comparison
```

性能比定义为：

```text
ratio = reference latency / candidate latency
```

同时记录不同 batch size、vocab size、top-k 和 dtype 下的变化，避免只依赖单一 shape 的结果。

## 目录

```text
src/h20_topk/__init__.py                      # candidate package
src/h20_topk/ops/compute_topk_scores.py       # candidate implementation
tests/test_compute_topk_scores.py             # correctness vs vLLM 0.29.0
benchmark/test_compute_topk_scores.py         # reference/candidate latency
scripts/bench_components.py                   # vLLM 各阶段耗时拆解
scripts/ablate_stats.py                       # scan/rank fusion 与 split 消融
scripts/bench_matrix.py                       # 多 workload 延迟矩阵
```

## 已验证环境（2026-10-24）

本机为 AutoDL 容器（Ubuntu 22.04.5，内核 5.15），当前为无 GPU 模式，以下版本来自实际环境：

| 项目 | 值 |
| --- | --- |
| Python | 3.12.13（`/root/autodl-tmp/venv-vllm029b`） |
| PyTorch | 2.13.0+cu130（`torch.version.cuda = 13.0`） |
| Triton | 3.7.1 |
| vLLM | 0.29.0（reference） |
| pytest | 9.1.1 |
| CUDA Toolkit | 13.0（`/usr/local/cuda` → `/usr/local/cuda-13.0`，nvcc V13.0.88） |
| pip cuda-toolkit | 13.0.3.0 |
| NVIDIA Driver | 595.71.05（宿主机，`/proc/driver/nvidia/version`） |
| GPU | NVIDIA H20（本机通过 `/proc/driver/nvidia/gpus` 可见） |
| CUDA available | False（当前无 GPU 模式） |

说明：

- 当前容器内 `/usr/bin/nvidia-smi` 是 0 字节且不可执行，`nvidia-smi` / `torch.cuda` 均不可用；驱动版本从 `/proc/driver/nvidia/version` 读取。
- 当前容器未安装 Docker 与 Conda；本仓库使用 Python venv + requirements，不依赖 Conda 环境文件。
- 当前无 GPU，未运行 GPU correctness / benchmark / profiler，不做性能结论。

## 新 H20 环境恢复

```bash
git clone https://github.com/WHUTcjh-2024/H20-CUDA-13.0-GPU-595.71.05.git
cd H20-CUDA-13.0-GPU-595.71.05

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.2
python -m pip install -r requirements/core.txt
```

验证：

```bash
python scripts/check_env.py
bash scripts/smoke_test.sh
```

无 GPU 时 `check_env.py` 仍以 0 退出；`smoke_test.sh` 会跳过 correctness/benchmark，设置 `REQUIRE_GPU=1` 可让缺失 GPU 直接失败。

candidate 直接从 `src/h20_topk` 导入，不需要额外的第三方 kernel 库。手工运行命令时可设置：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`requirements/freeze.txt` 是当前 venv 的完整审计快照，包含机器本地路径，不用于直接安装。

## 新 H20 测量顺序

先测三个 upstream kernel 的独立 baseline，再决定是否 fusion：

```bash
# 1. upstream vLLM 各阶段 baseline
python scripts/bench_components.py --output artifacts/baseline_components.json

# 2. correctness：reference 只使用 vLLM 0.29.0
QUICK_MODE=1 pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores

# 3. 再对比 candidate
python scripts/bench_matrix.py --target vllm --output artifacts/vllm.json
python scripts/bench_matrix.py --target candidate \
  --op src/h20_topk/ops/compute_topk_scores.py --output artifacts/candidate.json
```

## Docker

```bash
docker build -t h20-compute-topk:cu130 .
docker run --rm --gpus all h20-compute-topk:cu130
```

- 镜像只包含 CUDA userspace、Python、PyTorch、Triton 和 vLLM；**NVIDIA Driver 属于宿主机，不在镜像内**。
- 基础镜像固定为 `nvidia/cuda:13.0.0-devel-ubuntu24.04`，可通过 `--build-arg CUDA_IMAGE=...` 覆盖。
- 当前容器未安装 Docker，且数据盘仅剩约 1.1G；本次只做了 Dockerfile 静态检查，未实际 build/pull。

## 测试与 benchmark 入口

```bash
# correctness（需要 GPU）
QUICK_MODE=1 pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores

# latency benchmark（reference vs candidate）
pytest -q benchmark/test_compute_topk_scores.py -m compute_topk_scores
```

## 目录补充

```text
requirements/core.txt        # 最小核心依赖（固定版本）
requirements/freeze.txt      # 完整 pip freeze 审计快照
Dockerfile                   # CUDA 13.0 + vLLM 0.29.0 镜像
scripts/check_env.py         # 无 GPU 也安全的环境检查
scripts/smoke_test.sh        # 环境检查 -> import -> correctness -> benchmark
tests/conftest.py            # QUICK_MODE + src 加入 sys.path
src/h20_topk/__init__.py     # candidate package
```
