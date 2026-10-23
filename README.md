# compute_topk_scores — NVIDIA H20 算子优化工程

Reference: upstream `vllm==0.29.0`
`vllm/v1/worker/gpu/sample/logprob.py`。

Candidate: this repository (`src/h20_topk`).

Target GPU: NVIDIA H20.

## 结果

- Correctness：quick 57/57，full 129/129。
- Benchmark：72 个 case，串行测量，CUDA events，warmup=20，repeats=500，取 median。
- `ratio = reference latency / candidate latency`：

| 指标 | 值 |
| --- | ---: |
| ratio min | 0.980 |
| ratio p10 | 0.990 |
| ratio median | 1.172 |
| ratio mean | 1.149 |
| ratio p90 | 1.262 |
| ratio max | 1.280 |
| ratio < 0.8 | 0 |
| 0.8 <= ratio < 1.0 | 10 |
| ratio >= 1.0 | 62 |

全部 case 都 >= 0.98。10 个低于 1.0 的 case 都是 `V=32768`、`B=1/8`，ratio 在 `0.980–1.001`；大 vocab case 在 `1.09–1.28`。完整 72 行表格见
[`benchmark/results-h20.md`](benchmark/results-h20.md)。

按 B/V 聚合的中位 ratio：

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 0.986 | 1.236 | 1.262 |
| 8 | 0.992 | 1.251 | 1.273 |
| 64 | 1.043 | 1.192 | 1.197 |
| 256 | 1.045 | 1.158 | 1.169 |

## 范围

- **reference**：upstream `vllm==0.29.0` 的 `compute_topk_scores` 路径。
- **candidate**：本仓库实现，保持 upstream host path。
- `torch.topk` 是 upstream 组件，保留，不重写。
- 本仓库不依赖 FlagGems / FlagGems-vLLM；未来与 FlagGems-vLLM 的集成 **out of scope**。
- `src/h20_topk/ops/` 中只有这三个目标 kernel，没有额外的历史 kernel 文件。

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
优化全部放在三个 kernel body 内，host path 和 upstream launch config 保持不变：

| Kernel | Body change | 隔离延迟变化（bf16） |
| --- | --- | --- |
| `_topk_log_softmax_kernel` | online LSE + full-block/tail split + `tl.range(num_stages=2, loop_unroll_factor=2)` | 105→75us（V=131072）；210→120us（B=256） |
| `_ranks_kernel` | full-block/tail split + `tl.range(num_stages=4)` | 31→19.4us（V=131072）；67→37.9us（B=256） |
| `_fill_logprob_token_ids_kernel` | loads-first + 连续/对齐 hint | ~18.7→18.7us，launch-bound |

- fill 的 loads-first / write-through / no-custom 变体都在 18.6–18.8us，空 kernel 下限约 13.1us；
  剩余约 5.5us 是 token/mask 的实际访存。loads-first 作为最小 body 改动保留，结论是 **non-bottleneck**。
- 没有新增 Triton kernel，没有重写 `compute_topk_scores` host path，没有改 launch config。
- candidate 与 upstream 的差异只在三个 kernel body 和文件顶部 docstring。

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
  --dtypes bfloat16,float16 --warmup 20 --repeats 500 \
  --output artifacts/new-h20/vllm.json

python scripts/bench_matrix.py --target candidate \
  --op src/h20_topk/ops/compute_topk_scores.py \
  --batches 1,8,64,256 --vocabs 32768,131072,151936 --ks 1,10,50 \
  --dtypes bfloat16,float16 --warmup 20 --repeats 500 \
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
| Git HEAD（测量时） | `42ef89b` |

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
src/h20_topk/ops/compute_topk_scores.py       # candidate（三个目标 kernel）
tests/test_compute_topk_scores.py             # correctness vs vLLM 0.29.0
benchmark/test_compute_topk_scores.py         # reference/candidate latency
benchmark/results-h20.md                      # 完整 72-case 结果
scripts/bench_three_kernels.py                # 三个 upstream kernel 隔离 baseline
scripts/bench_kernel_variants.py              # launch-config 诊断（非主线）
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
