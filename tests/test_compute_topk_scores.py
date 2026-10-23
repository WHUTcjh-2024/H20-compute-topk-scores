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

"""Accuracy tests for ``flaggems_vllm.compute_topk_scores``.

The reference is vLLM's own ``compute_topk_scores``.  Ties are expected with
low-precision logits, so token ids are compared as a value multiset; ranks and
the sampled-token column are compared exactly.
"""

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

device = flaggems_vllm.device

try:
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores as _vllm_fn

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    _vllm_fn = None

pytestmark = pytest.mark.skipif(
    not HAS_VLLM or not torch.cuda.is_available(), reason="need vLLM + CUDA"
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

if cfg.QUICK_MODE:
    SHAPES = [(1, 1024), (8, 32768), (64, 131072)]
    KS = [0, 1, 5, 50]
else:
    SHAPES = [
        (1, 999),
        (1, 32768),
        (4, 65536),
        (16, 129280),
        (64, 131072),
        (256, 151936),
    ]
    KS = [0, 1, 5, 10, 20, 50]


def _make_inputs(batch, vocab, dtype, seed=0):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(batch, vocab, dtype=dtype, device=device, generator=generator)
    sampled = torch.randint(
        vocab, (batch,), dtype=torch.int64, device=device, generator=generator
    )
    return logits, sampled


def _row_values(logits, ids):
    return torch.gather(logits, 1, ids.long()).float()


def _sorted_rows(x):
    return torch.sort(x, dim=-1, descending=True).values


def _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids):
    assert out.logprob_token_ids.shape == ref.logprob_token_ids.shape
    assert out.logprobs.shape == ref.logprobs.shape
    assert out.selected_token_ranks.shape == ref.selected_token_ranks.shape
    assert out.logprob_token_ids.dtype == ref.logprob_token_ids.dtype
    assert out.logprobs.dtype == torch.float32
    assert out.selected_token_ranks.dtype == torch.int64

    assert torch.equal(out.logprob_token_ids[:, 0], sampled)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)

    torch.testing.assert_close(
        out.logprobs[:, 0], ref.logprobs[:, 0], atol=2e-3, rtol=1e-5
    )
    if num_logprobs == 0:
        return

    out_ids = out.logprob_token_ids[:, 1:]
    ref_ids = ref.logprob_token_ids[:, 1:]
    out_vals = _sorted_rows(_row_values(logits, out_ids))
    ref_vals = _sorted_rows(_row_values(logits, ref_ids))
    assert torch.equal(out_vals, ref_vals), "top-k values differ"

    torch.testing.assert_close(
        _sorted_rows(out.logprobs[:, 1:]),
        _sorted_rows(ref.logprobs[:, 1:]),
        atol=2e-3,
        rtol=1e-5,
    )
    if strict_ids:
        out_order = torch.argsort(_row_values(logits, out_ids), dim=-1, descending=True)
        ref_order = torch.argsort(_row_values(logits, ref_ids), dim=-1, descending=True)
        assert torch.equal(
            torch.gather(out_ids, 1, out_order), torch.gather(ref_ids, 1, ref_order)
        ), "top-k ids differ"


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_logprobs", KS)
@pytest.mark.parametrize("shape", SHAPES)
def test_compute_topk_scores_matches_vllm(shape, num_logprobs, dtype):
    batch, vocab = shape
    logits, sampled = _make_inputs(batch, vocab, dtype, seed=batch + vocab)
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(
        ref, out, logits, sampled, num_logprobs, strict_ids=(dtype == torch.float32)
    )


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_logprobs", [1, 5, 20])
def test_compute_topk_scores_logits_mode(dtype, num_logprobs):
    batch, vocab = 8, 65536
    logits, sampled = _make_inputs(batch, vocab, dtype, seed=7)
    ref = _vllm_fn(logits, num_logprobs, sampled, logits_mode=True)
    out = flaggems_vllm.compute_topk_scores(
        logits, num_logprobs, sampled, logits_mode=True
    )
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)
    assert torch.equal(out.logprob_token_ids[:, 0], sampled)
    torch.testing.assert_close(out.logprobs[:, 0], ref.logprobs[:, 0])
    out_vals = _sorted_rows(_row_values(logits, out.logprob_token_ids[:, 1:]))
    ref_vals = _sorted_rows(_row_values(logits, ref.logprob_token_ids[:, 1:]))
    assert torch.equal(out_vals, ref_vals)
    torch.testing.assert_close(
        _sorted_rows(out.logprobs[:, 1:]), _sorted_rows(ref.logprobs[:, 1:])
    )


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs", [1, 5, 50])
def test_compute_topk_scores_non_contiguous_rows(num_logprobs):
    batch, vocab, pad = 4, 32768, 128
    base, sampled = _make_inputs(batch, vocab + pad, torch.float32, seed=11)
    logits = base[:, :vocab]
    assert logits.stride(0) == vocab + pad
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids=True)


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs", [1, 10, 50])
def test_compute_topk_scores_ties(num_logprobs):
    batch, vocab = 8, 32768
    generator = torch.Generator(device=device)
    generator.manual_seed(3)
    logits = torch.randint(
        0, 8, (batch, vocab), device=device, generator=generator
    ).float()
    sampled = torch.randint(
        vocab, (batch,), dtype=torch.int64, device=device, generator=generator
    )
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids=False)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs,max_custom", [(5, 3), (4, 4), (0, 1)])
def test_compute_topk_scores_custom_token_ids(num_logprobs, max_custom):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

    batch, vocab = 4, 4096
    logits, sampled = _make_inputs(batch, vocab, torch.float32, seed=17)
    per_req = (
        [[1, 2, 3], [], [7], [10, 20]] if max_custom >= 3 else [[1], [], [7], [10]]
    )
    state = LogprobTokenIdsState(max_num_reqs=batch, device=torch.device("cuda"))
    for i, ids in enumerate(per_req):
        state.add_request(i, SamplingParams(logprob_token_ids=ids))
    state.apply_staged_writes()
    expanded_idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)

    kwargs = dict(
        logprob_token_ids_state=state,
        expanded_idx_mapping=expanded_idx_mapping,
        max_per_req_token_ids=max_custom,
    )
    ref = _vllm_fn(logits, num_logprobs, sampled, **kwargs)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled, **kwargs)
    assert torch.equal(out.logprob_token_ids, ref.logprob_token_ids)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)
    torch.testing.assert_close(out.logprobs, ref.logprobs, atol=2e-3, rtol=1e-5)


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_rejects_bad_inputs():
    logits, sampled = _make_inputs(2, 1024, torch.float32)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_scores(logits, 5, sampled.to(torch.int16))
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_scores(
            torch.empty((2, 0), device=device), 0, sampled
        )


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_empty_batch():
    logits = torch.empty((0, 1024), dtype=torch.float32, device=device)
    sampled = torch.empty((0,), dtype=torch.int64, device=device)
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled)
    assert out.logprob_token_ids.shape == (0, 6)
    assert out.logprobs.shape == (0, 6)
    assert out.selected_token_ranks.shape == (0,)


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_cu_num_logits_passthrough():
    logits, sampled = _make_inputs(4, 4096, torch.float32)
    cu_list = [0, 1, 2, 3, 4]
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled, cu_num_logits=cu_list)
    assert out.cu_num_generated_tokens == cu_list
    assert out.cu_num_generated_tokens_tensor is None

    cu_tensor = torch.tensor(cu_list, dtype=torch.int32, device=device)
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled, cu_num_logits=cu_tensor)
    assert out.cu_num_generated_tokens is None
    assert out.cu_num_generated_tokens_tensor is cu_tensor
