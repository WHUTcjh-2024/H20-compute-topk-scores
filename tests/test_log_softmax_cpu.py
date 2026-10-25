"""Check the masked-prefix regression with Triton's CPU interpreter."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.compute_topk_scores
def test_all_masked_prefix_with_triton_interpreter():
    # Triton reads TRITON_INTERPRET at import time; isolate it from pytest's
    # plugin imports and vLLM's no-GPU Triton placeholder.
    env = os.environ.copy()
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_interpreter_check():
    import torch
    import triton
    import triton.language as tl
    import vllm.triton_utils as vllm_triton_utils

    vllm_triton_utils.triton = triton
    vllm_triton_utils.tl = tl

    op_path = (
        Path(__file__).resolve().parents[1]
        / "src/h20_topk/ops/compute_topk_scores.py"
    )
    spec = importlib.util.spec_from_file_location("cpu_candidate_op", op_path)
    assert spec is not None and spec.loader is not None
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)

    from vllm.v1.worker.gpu.sample.logprob import (
        compute_topk_scores as reference_compute_topk_scores,
    )

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        vocab = 2064
        logits = torch.full((2, vocab), float("-inf"), dtype=dtype)
        logits[0, 1024:] = torch.linspace(-2, 2, vocab - 1024)
        logits[1, 2048:] = torch.arange(16, dtype=torch.float32)
        token_ids = torch.tensor([[1024, 2063], [2050, 2063]], dtype=torch.int64)

        actual = candidate.compute_token_logprobs(logits, token_ids)
        expected = torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids)
        assert torch.isfinite(actual).all(), dtype
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-5)

        sampled = torch.tensor([1024, 2050], dtype=torch.int64)
        output = candidate.compute_topk_scores(logits, 5, sampled)
        reference = reference_compute_topk_scores(logits, 5, sampled)
        assert torch.isfinite(output.logprobs).all(), dtype
        torch.testing.assert_close(
            output.logprobs, reference.logprobs, atol=2e-3, rtol=1e-5
        )
        assert torch.equal(output.logprob_token_ids, reference.logprob_token_ids)
        assert torch.equal(output.selected_token_ranks, reference.selected_token_ranks)


if __name__ == "__main__":
    _run_interpreter_check()

