# compute_topk_scores — NVIDIA H20 算子优化工程

Reference: upstream `vllm==0.29.0`
`vllm/v1/worker/gpu/sample/logprob.py`。

Candidate: this repository (`src/h20_topk`).

Target GPU: NVIDIA H20.

## 结果

- Correctness：quick 57/57，full 129/129。
- Benchmark：72 个 case，串行测量，CUDA events，warmup=20，repeats=200，取 median。
- `ratio = reference latency / candidate latency`：

| 指标 | 值 |
| --- | ---: |
| ratio min | 0.991 |
| ratio p10 | 1.002 |
| ratio median | 1.359 |
| ratio mean | 1.388 |
| ratio p90 | 1.805 |
| ratio max | 1.907 |
| ratio < 0.8 | 0 |
| 0.8 <= ratio < 1.0 | 5 |
| ratio >= 1.0 | 67 |

5 个低于 1.0 的 case 都是 `V=32768`，ratio 在 `0.991–0.996`，属于 launch config 与测量噪声范围。完整 72 行表格见
[`benchmark/results-h20.md`](benchmark/results-h20.md)。

按 B/V 聚合的中位 ratio：

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 1.004 | 1.733 | 1.802 |
| 8 | 0.998 | 1.754 | 1.833 |
| 64 | 1.014 | 1.474 | 1.483 |
| 256 | 1.006 | 1.228 | 1.245 |

## 范围

- **reference**：upstream `vllm==0.29.0` 的 `compute_topk_scores` 路径。
- **candidate**：本仓库实现，保持 upstream host path。
- `torch.topk` 是 upstream 组件，保留，不重写。
- 本仓库不依赖 FlagGems / FlagGems-vLLM；未来与 FlagGems-vLLM 的集成 **out of scope**。
- 旧 fused 实现保留在 `src/h20_topk/ops/fused_experiment.py`，仅作历史实验参考，不参与当前主线，也不被 package import。

## Upstream execution path

```text
torch.topk                         # upstream，保留
  -> _topk_log_softmax_kernel      # row max + sum-exp + gather
  -> _ranks_kernel                 # sampled token rank
  -> _fill_logprob_token_ids_kernel
```

host 侧的 `torch.topk`、`torch.cat`、`logits.gather`、`.to(torch.int32)`、
`torch.zeros_like`、`masked_fill` 均保持 upstream 写法。

## 优化内容

Baseline（bf16，隔离 kernel，preallocated output）：

| B | V | K | _topk_log_softmax | _ranks | _fill | torch.topk | full op |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 131072 | 10 | 106.80us | 31.46us | 18.53us | 58.11us | 214.10us |
| 8 | 131072 | 10 | 108.10us | 31.82us | 18.38us | 70.94us | 215.34us |
| 64 | 131072 | 10 | 108.94us | 32.24us | 18.21us | 161.26us | 292.19us |
| 256 | 151936 | 20 | 208.91us | 67.41us | 18.45us | 574.35us | 834.88us |

`_topk_log_softmax_kernel` 在 B<=64 时几乎恒为 ~107us，说明单 CTA 扫描是主要瓶颈。
先做串行 launch-config 粗筛，最终只改 launch config：

| Kernel | V <= 32768 | V > 32768 |
| --- | --- | --- |
| `_topk_log_softmax_kernel` | BLOCK_SIZE=1024, num_warps=4 | BLOCK_SIZE=8192, num_warps=16 |
| `_ranks_kernel` | BLOCK_SIZE=8192, num_warps=4 | BLOCK_SIZE=32768, num_warps=16 |
| `_fill_logprob_token_ids_kernel` | upstream | upstream |

- `_fill_logprob_token_ids_kernel` 在 1/2/4 warps 下都约 18.4us，属于 launch-bound，
  没有可观收益，保持 upstream。
- 没有新增 Triton kernel，没有重写 `compute_topk_scores` host path。
- candidate 与 upstream 的差异只在两个 kernel 的 launch config 和小 vocab 条件分支。

## Correctness

```bash
QUICK_MODE=1 pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores
pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores
```

- quick：57 passed
- full：129 passed

覆盖 fp16 / bf16 / float32、不同 B/V/K、non-contiguous、ties、custom token ids、
logits_mode、empty batch、cu_num_logits。

## Benchmark

串行执行，reference 与 candidate 使用相同输入、dtype、B/V/K、warmup 和 repeats：

```bash
python scripts/bench_matrix.py --target vllm \
  --batches 1,8,64,256 --vocabs 32768,131072,151936 --ks 1,10,50 \
  --dtypes bfloat16,float16 --warmup 20 --repeats 200 \
  --output artifacts/new-h20/vllm.json

python scripts/bench_matrix.py --target candidate \
  --op src/h20_topk/ops/compute_topk_scores.py \
  --batches 1,8,64,256 --vocabs 32768,131072,151936 --ks 1,10,50 \
  --dtypes bfloat16,float16 --warmup 20 --repeats 200 \
  --output artifacts/new-h20/candidate.json

python scripts/compare_bench.py \
  --reference artifacts/new-h20/vllm.json \
  --candidate artifacts/new-h20/candidate.json
```

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA H20 |
| NVIDIA Driver | 580.105.08（宿主机） |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| vLLM | 0.29.0（reference） |
| Python | 3.12.3 |
| Git HEAD（测量时） | `afc64fd` |

## 复现

```bash
git clone https://github.com/WHUTcjh-2024/H20-compute-topk-scores.git
cd H20-compute-topk-scores

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.2
python -m pip install -r requirements/core.txt

python scripts/check_env.py
bash scripts/smoke_test.sh
```

candidate 直接从 `src/h20_topk` 导入，不需要额外第三方 kernel 库：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`requirements/freeze.txt` 是完整 pip freeze 审计快照，包含机器本地路径，不用于直接安装。

## 目录

```text
src/h20_topk/ops/compute_topk_scores.py       # candidate
src/h20_topk/ops/fused_experiment.py          # 历史 fused 实验，不参与主线
tests/test_compute_topk_scores.py             # correctness vs vLLM 0.29.0
benchmark/test_compute_topk_scores.py         # reference/candidate latency
benchmark/results-h20.md                      # 完整 72-case 结果
scripts/bench_three_kernels.py                # 三个 upstream kernel 隔离 baseline
scripts/bench_kernel_variants.py              # launch-config 串行筛选
scripts/bench_matrix.py                       # 多 workload 延迟矩阵
scripts/compare_bench.py                      # reference/candidate ratio 汇总
scripts/bench_components.py                   # vLLM 各阶段耗时拆解
scripts/check_env.py                          # 环境检查
scripts/smoke_test.sh                         # 环境 -> import -> correctness -> benchmark
```

## Docker

```bash
docker build -t h20-compute-topk:cu130 .
docker run --rm --gpus all h20-compute-topk:cu130
```

- 镜像只包含 CUDA userspace、Python、PyTorch、Triton 和 vLLM；
  **NVIDIA Driver 属于宿主机，不在镜像内**。
- 基础镜像固定为 `nvidia/cuda:13.0.0-devel-ubuntu24.04`，可用
  `--build-arg CUDA_IMAGE=...` 覆盖。
- 本次测量容器未安装 Docker，因此只做了 Dockerfile 静态检查，未实际 build/pull。
