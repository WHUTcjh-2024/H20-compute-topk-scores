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

"""Performance benchmark for the optimized ``compute_topk_scores`` candidate.

Baseline: vLLM's own ``compute_topk_scores``.
Candidate: the fused-scan implementation that deliberately keeps ``torch.topk``.
No native/radix top-k implementation is part of this repository scope.
"""

import pytest
import torch

import flaggems_vllm

from . import base, consts

vendor_name = flaggems_vllm.vendor_name


class ComputeTopkScoresBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name, torch_op, dtypes)
        self.gems_op = None
        self.shape_desc = "B, V, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 32768, 0),
            (1, 32768, 10),
            (1, 131072, 10),
            (8, 131072, 50),
            (64, 65536, 10),
            (64, 131072, 10),
            (256, 32768, 0),
            (256, 151936, 20),
        ]

    def get_input_iter(self, dtype):
        for batch, vocab, topk in self.shapes:
            logits = torch.randn(batch, vocab, dtype=dtype, device=self.device)
            sampled = torch.randint(vocab, (batch,), device=self.device)
            yield logits, topk, sampled

    def set_gems(self, gems_op):
        self.gems_op = gems_op


@pytest.mark.compute_topk_scores
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_topk_scores():
    vllm_logprob = pytest.importorskip("vllm.v1.worker.gpu.sample.logprob")

    bench = ComputeTopkScoresBenchmark(
        op_name="compute_topk_scores",
        torch_op=vllm_logprob.compute_topk_scores,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flaggems_vllm.compute_topk_scores)
    bench.run()
