"""Kernel replacement boundaries for future optimized merging."""

from .fused_merger import fused_triton_merge_tokens
from .reference import reference_merge

__all__ = ["fused_triton_merge_tokens", "reference_merge"]
