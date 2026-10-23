# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance benchmark for the candidate ``compute_topk_scores``.

Reference: vLLM 0.29.0 ``vllm.v1.worker.gpu.sample.logprob.compute_topk_scores``.
Candidate: this repository (``h20_topk``).
"""

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h20_topk import compute_topk_scores as candidate_fn

SHAPES = [
    (1, 32768, 0),
    (1, 32768, 10),
    (1, 131072, 10),
    (8, 131072, 50),
    (64, 65536, 10),
    (64, 131072, 10),
    (256, 32768, 0),
    (256, 151936, 20),
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _median_ms(fn, args, warmup=10, repeats=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


@pytest.mark.compute_topk_scores
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
def test_compute_topk_scores(dtype):
    logprob = pytest.importorskip("vllm.v1.worker.gpu.sample.logprob")
    reference_fn = logprob.compute_topk_scores

    for batch, vocab, topk in SHAPES:
        torch.manual_seed(314159 + batch + vocab + topk)
        logits = torch.randn(batch, vocab, dtype=dtype, device="cuda")
        sampled = torch.randint(vocab, (batch,), device="cuda")
        args = (logits, topk, sampled)

        ref = reference_fn(*args)
        cand = candidate_fn(*args)
        assert cand.logprob_token_ids.shape == ref.logprob_token_ids.shape
        assert cand.logprobs.shape == ref.logprobs.shape
        assert cand.selected_token_ranks.shape == ref.selected_token_ranks.shape

        ref_ms = _median_ms(reference_fn, args)
        cand_ms = _median_ms(candidate_fn, args)
        print(
            f"[bench] {dtype} B={batch} V={vocab} K={topk} "
            f"reference={ref_ms:.4f}ms candidate={cand_ms:.4f}ms "
            f"ratio={ref_ms / cand_ms:.3f}x"
        )
