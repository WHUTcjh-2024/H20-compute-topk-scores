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

"""Triton implementation of vLLM's ``compute_topk_scores``.

Phase 1 keeps ``torch.topk`` for the ranking/selection step and replaces the
row-scanning kernels:

* vLLM runs ``_topk_log_softmax_kernel`` (row max, then row sum-exp, then a
  gather) and ``_ranks_kernel`` (another full row scan), for three passes over
  the logits.  Here one streaming kernel computes the row max, the log-sum-exp
  (online) and the sampled-token rank in a single pass.  Each row is split
  across several programs when the batch is small so the GPU is filled.
* A second, per-row kernel combines the partials and emits the log-softmax
  scores of the selected ids, so no separate score kernel is launched.  When a
  single program owns the row, scan and finalize are fused into one kernel.

Semantics match ``vllm.v1.worker.gpu.sample.logprob.compute_topk_scores``
(vLLM 0.29.0).  The ``torch.topk`` call is retained deliberately as the
phase-1 baseline and is the next replacement target.
"""

import torch
import triton
import triton.language as tl

_MAX_SPLITS = 32
_TARGET_CTAS = 1024
_SCORE_BLOCK = 128


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_rank_ptr,
    vocab_size,
    num_tiles,
    tile_span,
    COMPUTE_LSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * tile_span
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    out_idx = row * tl.num_programs(1) + split

    token = tl.load(sampled_ptr + row).to(tl.int64)
    token_ok = (token >= 0) & (token < vocab_size)
    sampled_val = tl.load(row_ptr + token, mask=token_ok, other=float("inf")).to(
        tl.float32
    )

    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    for tile in tl.range(0, num_tiles):
        idx = start + tile * BLOCK + tl.arange(0, BLOCK)
        mask = idx < vocab_size
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if COMPUTE_LSE:
            tile_max = tl.max(x)
            new_max = tl.maximum(running_max, tile_max)
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            contrib = tl.where(new_max == -float("inf"), 0.0, tl.exp(x - new_max))
            contrib = tl.where(mask, contrib, 0.0)
            running_sum = running_sum * scale + tl.sum(contrib)
            running_max = new_max
        rank += tl.sum(((x >= sampled_val) & mask).to(tl.int32))

    tl.store(partial_max_ptr + out_idx, running_max)
    tl.store(partial_sum_ptr + out_idx, running_sum)
    tl.store(partial_rank_ptr + out_idx, rank)


@triton.jit
def _single_row_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    topk_ids_ptr,
    lse_ptr,
    rank_ptr,
    out_ids_ptr,
    out_scores_ptr,
    vocab_size,
    num_tiles,
    topk,
    COMPUTE_LSE: tl.constexpr,
    LOGITS_MODE: tl.constexpr,
    OUT_I64: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Whole-row scan plus finalize, used when one program owns a row."""
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride

    token = tl.load(sampled_ptr + row).to(tl.int64)
    token_ok = (token >= 0) & (token < vocab_size)
    sampled_val = tl.load(row_ptr + token, mask=token_ok, other=float("inf")).to(
        tl.float32
    )

    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    for tile in tl.range(0, num_tiles):
        idx = tile * BLOCK + tl.arange(0, BLOCK)
        mask = idx < vocab_size
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if COMPUTE_LSE:
            tile_max = tl.max(x)
            new_max = tl.maximum(running_max, tile_max)
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            contrib = tl.where(new_max == -float("inf"), 0.0, tl.exp(x - new_max))
            contrib = tl.where(mask, contrib, 0.0)
            running_sum = running_sum * scale + tl.sum(contrib)
            running_max = new_max
        rank += tl.sum(((x >= sampled_val) & mask).to(tl.int32))

    if COMPUTE_LSE:
        lse = running_max + tl.log(running_sum)
    else:
        lse = 0.0
    tl.store(lse_ptr + row, lse)
    tl.store(rank_ptr + row, rank.to(tl.int64))

    if WRITE_SCORES:
        sampled = tl.load(sampled_ptr + row)
        for base in tl.range(0, topk + 1, BLOCK):
            cols = base + tl.arange(0, BLOCK)
            mask = cols < topk + 1
            topk_id = tl.load(
                topk_ids_ptr + row * topk + cols - 1, mask=mask & (cols > 0), other=0
            )
            ids = tl.where(cols == 0, sampled.to(tl.int64), topk_id.to(tl.int64))
            x = tl.load(row_ptr + ids, mask=mask, other=0.0).to(tl.float32)
            if LOGITS_MODE:
                score = x
            else:
                score = x - lse
            out_idx = row * (topk + 1) + cols
            if OUT_I64:
                tl.store(out_ids_ptr + out_idx, ids, mask=mask)
            else:
                tl.store(out_ids_ptr + out_idx, ids.to(tl.int32), mask=mask)
            tl.store(out_scores_ptr + out_idx, score, mask=mask)


@triton.jit
def _finalize_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_rank_ptr,
    logits_ptr,
    logits_stride,
    sampled_ptr,
    topk_ids_ptr,
    lse_ptr,
    rank_ptr,
    out_ids_ptr,
    out_scores_ptr,
    num_splits,
    topk,
    COMPUTE_LSE: tl.constexpr,
    LOGITS_MODE: tl.constexpr,
    OUT_I64: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    for split in tl.range(0, num_splits):
        out_idx = row * num_splits + split
        rank += tl.load(partial_rank_ptr + out_idx)
        if COMPUTE_LSE:
            part_max = tl.load(partial_max_ptr + out_idx)
            part_sum = tl.load(partial_sum_ptr + out_idx)
            new_max = tl.maximum(running_max, part_max)
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            grow = tl.where(new_max == -float("inf"), 0.0, tl.exp(part_max - new_max))
            running_sum = running_sum * scale + part_sum * grow
            running_max = new_max

    if COMPUTE_LSE:
        lse = running_max + tl.log(running_sum)
    else:
        lse = 0.0
    tl.store(lse_ptr + row, lse)
    tl.store(rank_ptr + row, rank.to(tl.int64))

    if WRITE_SCORES:
        sampled = tl.load(sampled_ptr + row)
        row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
        for base in tl.range(0, topk + 1, BLOCK):
            cols = base + tl.arange(0, BLOCK)
            mask = cols < topk + 1
            topk_id = tl.load(
                topk_ids_ptr + row * topk + cols - 1, mask=mask & (cols > 0), other=0
            )
            ids = tl.where(cols == 0, sampled.to(tl.int64), topk_id.to(tl.int64))
            x = tl.load(row_ptr + ids, mask=mask, other=0.0).to(tl.float32)
            if LOGITS_MODE:
                score = x
            else:
                score = x - lse
            out_idx = row * (topk + 1) + cols
            if OUT_I64:
                tl.store(out_ids_ptr + out_idx, ids, mask=mask)
            else:
                tl.store(out_ids_ptr + out_idx, ids.to(tl.int32), mask=mask)
            tl.store(out_scores_ptr + out_idx, score, mask=mask)


@triton.jit
def _fill_custom_ids_kernel(
    out_token_ids_ptr,
    out_token_ids_stride,
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_ptr,
    topk_ids_ptr,
    topk_ids_stride,
    expanded_idx_mapping_ptr,
    num_custom_ptr,
    custom_ids_ptr,
    custom_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_I64: tl.constexpr,
):
    """Adapted from vLLM 0.29.0 ``_fill_logprob_token_ids_kernel``."""
    row = tl.program_id(0)

    sampled = tl.load(sampled_ptr + row)
    if OUT_I64:
        tl.store(out_token_ids_ptr + row * out_token_ids_stride, sampled.to(tl.int64))
    else:
        tl.store(out_token_ids_ptr + row * out_token_ids_stride, sampled.to(tl.int32))
    tl.store(out_valid_mask_ptr + row * out_valid_mask_stride, 1)

    req_idx = tl.load(expanded_idx_mapping_ptr + row)
    num_custom = tl.load(num_custom_ptr + req_idx)

    col = tl.arange(0, PADDED_COLS)
    ids_base = out_token_ids_ptr + row * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + row * out_valid_mask_stride + 1

    # Default every padded column to a valid, harmless (id 0, invalid) entry so
    # the gather in the score kernel never sees uninitialized ids.  ``ids_base``
    # already points at column 1, so the row holds OUT_WIDTH - 1 of them.
    in_row = col < OUT_WIDTH - 1
    if OUT_I64:
        tl.store(ids_base + col, tl.zeros([PADDED_COLS], dtype=tl.int64), mask=in_row)
    else:
        tl.store(ids_base + col, tl.zeros([PADDED_COLS], dtype=tl.int32), mask=in_row)
    tl.store(mask_base + col, tl.zeros([PADDED_COLS], dtype=tl.int1), mask=in_row)

    if num_custom > 0:
        src = custom_ids_ptr + req_idx * custom_ids_stride
        valid = col < num_custom
    else:
        src = topk_ids_ptr + row * topk_ids_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    if OUT_I64:
        tl.store(ids_base + col, tokens, mask=valid)
    else:
        tl.store(ids_base + col, tokens.to(tl.int32), mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


@triton.jit
def _custom_scores_kernel(
    logits_ptr,
    logits_stride,
    ids_ptr,
    valid_mask_ptr,
    lse_ptr,
    scores_ptr,
    num_cols,
    LOGITS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < num_cols
    ids = tl.load(ids_ptr + row * num_cols + cols, mask=mask, other=0).to(tl.int64)
    valid = tl.load(valid_mask_ptr + row * num_cols + cols, mask=mask, other=0)
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + ids, mask=mask, other=0.0
    ).to(tl.float32)
    if LOGITS_MODE:
        score = x
    else:
        score = x - tl.load(lse_ptr + row)
    score = tl.where(valid, score, float("-inf"))
    tl.store(scores_ptr + row * num_cols + cols, score, mask=mask)


def _choose_splits(batch_size, vocab_size, topk):
    """Programs per row.

    Small vocabularies are cheap enough that fusing scan and finalize into one
    program per row beats paying a second launch; wider rows need the vocab
    split to fill the GPU.  Without top-k there is much less work, so the fused
    path stays preferable up to a larger batch.  Thresholds calibrated on H20.
    """
    if batch_size <= 0:
        return 1
    if vocab_size <= 32768:
        if batch_size <= 64 or (batch_size <= 128 and topk == 0):
            return 1
        return 4
    target = min(_MAX_SPLITS, max(2, _TARGET_CTAS // batch_size))
    return max(1, min(target, vocab_size // 8192))


def _run_rows(
    logits,
    sampled_token_ids,
    topk_ids,
    out_ids,
    scores,
    logits_mode,
    out_i64,
    write_scores,
):
    """Run the row scan (fused or split) and return lse and token ranks."""
    batch_size, vocab_size = logits.shape
    device = logits.device
    topk = topk_ids.shape[1] if topk_ids is not None else 0
    topk_arg = topk_ids if topk_ids is not None else sampled_token_ids
    splits = _choose_splits(batch_size, vocab_size, topk)

    lse = torch.empty((batch_size,), dtype=torch.float32, device=device)
    ranks = torch.empty((batch_size,), dtype=torch.int64, device=device)

    if splits == 1:
        block = 8192 if vocab_size >= 8192 else 4096
        _single_row_kernel[(batch_size,)](
            logits,
            logits.stride(0),
            sampled_token_ids,
            topk_arg,
            lse,
            ranks,
            out_ids,
            scores,
            vocab_size,
            triton.cdiv(vocab_size, block),
            topk,
            COMPUTE_LSE=not logits_mode,
            LOGITS_MODE=logits_mode,
            OUT_I64=out_i64,
            WRITE_SCORES=write_scores,
            BLOCK=block,
            num_warps=8,
        )
        return lse, ranks

    chunk = triton.cdiv(vocab_size, splits)
    block = 8192 if chunk >= 8192 else 4096
    tile_span = triton.cdiv(chunk, block) * block

    partial_max = torch.empty((batch_size, splits), dtype=torch.float32, device=device)
    partial_sum = torch.empty((batch_size, splits), dtype=torch.float32, device=device)
    partial_rank = torch.empty((batch_size, splits), dtype=torch.int32, device=device)
    _row_stats_kernel[(batch_size, splits)](
        logits,
        logits.stride(0),
        sampled_token_ids,
        partial_max,
        partial_sum,
        partial_rank,
        vocab_size,
        triton.cdiv(chunk, block),
        tile_span,
        COMPUTE_LSE=not logits_mode,
        BLOCK=block,
        num_warps=8,
    )
    _finalize_kernel[(batch_size,)](
        partial_max,
        partial_sum,
        partial_rank,
        logits,
        logits.stride(0),
        sampled_token_ids,
        topk_arg,
        lse,
        ranks,
        out_ids,
        scores,
        splits,
        topk,
        COMPUTE_LSE=not logits_mode,
        LOGITS_MODE=logits_mode,
        OUT_I64=out_i64,
        WRITE_SCORES=write_scores,
        BLOCK=_SCORE_BLOCK,
    )
    return lse, ranks


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
):
    """Return the top-k logprobs of ``logits`` for the sampled tokens.

    Matches vLLM's ``compute_topk_scores`` contract.
    """
    from vllm.v1.outputs import LogprobsTensors

    assert num_logprobs >= 0
    if logits.dim() != 2:
        raise NotImplementedError("compute_topk_scores expects 2D logits")
    if logits.stride(-1) != 1:
        raise NotImplementedError("only contiguous rows are supported")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"unsupported logits dtype: {logits.dtype}")
    if sampled_token_ids.numel() != logits.shape[0]:
        raise ValueError("sampled_token_ids must have one entry per row")
    if sampled_token_ids.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError(
            f"unsupported sampled_token_ids dtype: {sampled_token_ids.dtype}"
        )
    if sampled_token_ids.device != logits.device:
        raise ValueError("logits and sampled_token_ids must be on one device")

    batch_size, vocab_size = logits.shape
    if vocab_size == 0:
        raise NotImplementedError("compute_topk_scores requires vocab_size > 0")
    device = logits.device
    out_i64 = sampled_token_ids.dtype == torch.int64

    if batch_size == 0:
        width = 1 + max(num_logprobs, max_per_req_token_ids)
        return LogprobsTensors(
            logprob_token_ids=torch.empty(
                (0, width), dtype=sampled_token_ids.dtype, device=device
            ),
            logprobs=torch.empty((0, width), dtype=torch.float32, device=device),
            selected_token_ranks=torch.empty((0,), dtype=torch.int64, device=device),
            cu_num_generated_tokens=(
                None if isinstance(cu_num_logits, torch.Tensor) else cu_num_logits
            ),
            cu_num_generated_tokens_tensor=(
                cu_num_logits if isinstance(cu_num_logits, torch.Tensor) else None
            ),
        )

    # Phase 1: selection stays on torch.topk (upstream baseline path).
    topk_ids = None
    if num_logprobs > 0:
        topk_ids = torch.topk(logits, num_logprobs, dim=-1).indices

    if max_per_req_token_ids == 0:
        width = num_logprobs + 1
        out_ids = torch.empty(
            (batch_size, width), dtype=sampled_token_ids.dtype, device=device
        )
        scores = torch.empty((batch_size, width), dtype=torch.float32, device=device)
        _, ranks = _run_rows(
            logits,
            sampled_token_ids,
            topk_ids,
            out_ids,
            scores,
            logits_mode,
            out_i64,
            True,
        )
    else:
        if logprob_token_ids_state is None or expanded_idx_mapping is None:
            raise ValueError(
                "logprob_token_ids_state and expanded_idx_mapping are required "
                "when max_per_req_token_ids > 0"
            )
        num_cols = max(num_logprobs, max_per_req_token_ids)
        width = num_cols + 1
        state_ids = logprob_token_ids_state.token_ids.gpu
        # The fill kernel takes int32 ids on both branches, matching vLLM.
        topk_i32 = topk_ids.to(torch.int32) if topk_ids is not None else state_ids
        out_ids = torch.empty(
            (batch_size, width), dtype=sampled_token_ids.dtype, device=device
        )
        valid_mask = torch.empty((batch_size, width), dtype=torch.bool, device=device)
        scores = torch.empty((batch_size, width), dtype=torch.float32, device=device)
        lse, ranks = _run_rows(
            logits,
            sampled_token_ids,
            topk_ids,
            out_ids,
            scores,
            logits_mode,
            out_i64,
            False,
        )
        _fill_custom_ids_kernel[(batch_size,)](
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_i32,
            topk_i32.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            state_ids,
            state_ids.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(width),
            OUT_WIDTH=width,
            OUT_I64=out_i64,
        )
        _custom_scores_kernel[(batch_size, triton.cdiv(width, _SCORE_BLOCK))](
            logits,
            logits.stride(0),
            out_ids,
            valid_mask,
            lse,
            scores,
            width,
            LOGITS_MODE=logits_mode,
            BLOCK=_SCORE_BLOCK,
        )

    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=out_ids,
        logprobs=scores,
        selected_token_ranks=ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )
