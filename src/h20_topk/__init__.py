"""Candidate Triton implementation of vLLM's ``compute_topk_scores``."""

from .ops.compute_topk_scores import compute_topk_scores

__all__ = ["compute_topk_scores"]
__version__ = "0.1.0"
