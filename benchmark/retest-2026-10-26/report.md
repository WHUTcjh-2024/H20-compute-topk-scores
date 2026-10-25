# H20 `compute_topk_scores` 复测报告（2026-10-26）

> 参考 FlagGems-vllm `workflow.md` 的项目真相、契约、路径、验证、性能和交付摘要栏目组织本报告；验收对象仅为本独立仓库的 vLLM kernel-body 优化。

## 结论

- **本仓库复测达标。** 当前 `main` 提交 `d6c980c87c27a6bd62492194e42c6a6cfd923e1a` 在新宿主机 H20 上通过 CPU 1/1、GPU quick 60/60、GPU full 132/132。72 个 active case 的 `SpeedUp = vLLM reference latency / candidate latency` 最小为 **0.982**，高于本报告参考的关键 shape 门槛 **0.9**。
- 7/72 个 case 略慢于 reference，全部为 `V=32768`；最差 0.982（约慢 1.8%）。这是单次串行复测，不能据此断言这些小幅差异稳定；也没有 case 低于 0.9。

## 任务与项目真相

| 字段 | 本次冻结值 |
| --- | --- |
| Op / reference | `h20_topk.compute_topk_scores` 对照 vLLM 0.29.0 `vllm.v1.worker.gpu.sample.logprob.compute_topk_scores` |
| 测试提交 | `d6c980c87c27a6bd62492194e42c6a6cfd923e1a`；测试前后工作树干净 |
| 本轮可编辑 | 本报告、README 索引和复测证据；不改候选 kernel、reference、测试或 benchmark 脚本 |
| Read-only | `src/h20_topk/`、`tests/`、`scripts/bench_matrix.py`、vLLM 0.29.0 reference 和旧 `benchmark/results-h20.*` |
| 构建 | 纯 Python 源码直接导入，无本地编译步骤；独立 `.venv` 按 `requirements/core.txt` 安装 |
| Validation | `tests/test_log_softmax_cpu.py`；`tests/test_compute_topk_scores.py` quick/full，以全部通过为准 |
| Benchmark | `scripts/bench_matrix.py`，CUDA event 每调用同步；每 case 的 median 延迟（µs，越低越好）及 reference/candidate 比值（越高越好） |
| Active set | dtype={bf16, fp16} × B={1,8,64,256} × V={32768,131072,151936} × K={1,10,50}，共 72 case |
| Aggregation | 先逐 case 计算比值，再报告 min、p10、median、mean、p90、max；最低 case 是硬门槛检查，不用平均值掩盖回归 |
| 测量设置 | reference 与 candidate 各自独立进程、相同固定随机种子，串行执行；warmup=20，repeats=500 |
| Timeout | 本轮未设置测试/benchmark 超时；全部命令正常结束，不能将其解释为超时压力测试 |
| Profiler | `nsys` 可用、`ncu` 未找到；本轮只复测既有实现，未做新优化或 profiler 归因 |
| Autotune | 本项目只改三个 kernel body，沿用 upstream 固定 launch 参数；本轮未调参 |
| Fallback policy | 沿用 upstream vLLM host path，没有新增 fallback、替换 reference 或改变 dispatch |

环境来自 `environment.json`，`pip check` 与 `check_env.py --strict --require-gpu` 均通过。GPU 为 H20（SM 9.0，97871 MiB），驱动 580.65.06，CUDA toolkit / PyTorch CUDA 均为 13.0；Python 3.12.15、PyTorch 2.13.0+cu130、Triton 3.7.1、vLLM 0.29.0。旧 Conda 环境的 vLLM 0.20.2 未参与复测。

## 契约与实现路径

| 项目 | 本次已验证的范围与限制 |
| --- | --- |
| 输入 | CUDA logits `[B,V]`；主测试覆盖 fp16/bf16/fp32、B≤256、V≤151936、K=0/1/5/10/20/50；测试包括非连续行 stride |
| 输出 | token id 列 0 是 sampled token；常规输出为 `[B,K+1]`，score 为 fp32，rank 为 int64；按 vLLM reference 对拍 |
| 语义 | 覆盖 ties、`logits_mode`、custom token ids、`cu_num_logits`、空 batch、首段全 `-inf` 且后段含有限值 |
| 未验证 | autograd/训练、多 GPU、其他硬件、全部可能 dtype/layout；empty vocab 在 upstream 0.29.0 也可能触发非法 CUDA 访问，测试明确未覆盖 |
| Fast / general | 无 custom ids 时沿用 vLLM host path：`torch.topk` 生成候选，Triton log-softmax 与 rank kernel 完成目标计算 |
| Special | custom token ids 走 Triton fill kernel；`logits_mode`、空 batch 和 metadata 传递有测试 |
| External library | `torch.topk` 是保留的 upstream 组件，不计作自研 Triton kernel 收益 |

**Host path / fallback 核对：** `src/h20_topk/ops/compute_topk_scores.py` 保留 upstream 的 `torch.topk`、`torch.cat`、`gather` 等 host 计算；本项目测的是完整 vLLM 路径下三个目标 Triton kernel body 的改动。上述调用不是本轮新增的 fallback，也不计入自研 kernel 收益。本轮没有改 host dispatch；kernel launch 仍固定 `BLOCK_SIZE=1024/8192`，未引入 autotune。

## 功能测试

| 命令 | 当前结果 |
| --- | ---: |
| `.venv/bin/python -m pytest -q tests/test_log_softmax_cpu.py` | 1 passed |
| `QUICK_MODE=1 .venv/bin/python -m pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores` | 60 passed |
| `.venv/bin/python -m pytest -q tests/test_compute_topk_scores.py -m compute_topk_scores` | 132 passed |

quick/full 比旧 README 的 57/129 各多 3 个用例，包含本提交新增的 `-inf` 前缀测试。对应原始输出保存在 `tests-cpu.txt`、`tests-quick.txt`、`tests-full.txt`。

## 性能验收

同一脚本 SHA-256 为 `bc67650cb88fd1eb7324f5fa8c755ed2eeea798e16265f8d52ea4ff4ab528db2`；reference/candidate 的 JSON 各含 72 个唯一且完全匹配的 case，环境和提交一致。原始逐 case 数据见 `vllm.json`、`candidate.json` 和 `comparison.json`。

| 指标 | 新宿主机实测 |
| --- | ---: |
| SpeedUp min / p10 / median | **0.982 / 0.995 / 1.163** |
| SpeedUp mean / p90 / max | 1.148 / 1.272 / 1.297 |
| `SpeedUp < 0.8` | 0 / 72 |
| `0.8 ≤ SpeedUp < 1.0` | 7 / 72 |
| `SpeedUp ≥ 1.0` | 65 / 72 |
| `SpeedUp < 0.9` | **0 / 72** |

| B | V=32768 | V=131072 | V=151936 |
| ---: | ---: | ---: | ---: |
| 1 | 1.006 | 1.254 | 1.276 |
| 8 | 0.999 | 1.260 | 1.282 |
| 64 | 1.013 | 1.186 | 1.190 |
| 256 | 1.035 | 1.152 | 1.162 |

上表为每个 B/V 内 6 个 dtype/K case 的**比值中位数**。最差 case：fp16、B=8、V=32768、K=1，reference **115.84 µs**，candidate **117.94 µs**，SpeedUp **0.982**。最大收益：bf16、B=1、V=151936、K=1，SpeedUp **1.297**。7 个低于 1.0 的 case 全在 V=32768；原始逐 case 表没有被平均值替代。

**验收判定：** 以 `workflow.md` §7 默认的关键 shape `SpeedUp ≥ 0.9` 为数值门槛，72/72 均通过。小 shape 的 0.6%–1.8% 劣势需要后续重复测量才能判断稳定性；本轮没有证据表明存在低于 0.9 的关键回归。历史 `42ef89b` 报告使用驱动 580.105.08、Python 3.12.3，而本次提交及驱动均不同；旧绝对时延不作为同环境对照。

## Operator delivery summary

- **Op：** vLLM 0.29.0 `compute_topk_scores` / 独立包 `h20_topk.compute_topk_scores`。
- **Status：** 当前提交在本项目定义的 H20 复测范围内 `complete`，性能达到本报告采用的门槛。
- **Torch contract：** 见契约表；本轮仅对已列出的 dtype、shape、布局及边界进行对拍。
- **Paths：** fast/general、custom-id special 和 `logits_mode` 见契约与路径表；empty vocab 不在支持性声明中。
- **Host path / fallback：** 保留 vLLM upstream 的 torch host 计算；没有新增 fallback 或更改 dispatch。
- **Autotune：** 未接入；本项目保持 upstream 固定 launch 参数，只比较 kernel body。
- **Tests：** CPU 1/1、GPU quick 60/60、GPU full 132/132，均在当前提交 H20 上通过。
- **Benchmark：** 72 个 active case；min 0.982、median 1.163，7 个略低于 1.0，0 个低于 0.9。
- **Deep opt：** 按数值门槛未触发；本轮无新策略、无 keep/revert 决策。若后续要求所有 case `≥1.0`，应先冻结新目标和 baseline，再用重复测量确认小 shape 差异。
- **Residual risks：** 单次串行测量的小幅差异、无当前提交的 kernel 分解/profiler 数据、empty vocab、未覆盖的 dtype/layout/autograd、跨宿主机绝对时延不可直接比较。

## 复测检查表（借用 workflow 栏目）

| Gate | 结论 |
| --- | --- |
| G0 任务输入 | pass：目标 API、vLLM 0.29.0 reference 和 H20 已固定 |
| G1 项目真相 | pass：提交、环境、active set、计量与证据已冻结 |
| G2 契约 | pass（本项目范围）：对拍已列明的语义，未测范围单独记录 |
| G3 路径设计 | pass（本项目范围）：保持 vLLM host path，三个目标 kernel body 为候选差异 |
| G4 实现集成 | pass（本项目范围）：独立包导出、测试和 benchmark 已具备；本轮未改代码 |
| G5 功能测试 | pass：CPU 1/1，GPU quick 60/60，full 132/132 |
| G6 性能验收 | pass：所有 72 个 active case ≥0.982，超过 0.9 门槛 |
| G7 交付 | pass：报告、环境记录、测试输出和 72 case 原始 JSON 完整 |
