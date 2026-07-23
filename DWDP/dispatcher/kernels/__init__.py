"""Kernel entry points for future fused dispatch implementations."""

from .reference import (
    counting_scatter_expert_major_dispatch,
    reference_expert_major_dispatch,
    stable_sort_expert_major_dispatch,
)
from .triton import TRITON_AVAILABLE, triton_counting_scatter_expert_major_dispatch

__all__ = [
    "TRITON_AVAILABLE",
    "counting_scatter_expert_major_dispatch",
    "reference_expert_major_dispatch",
    "stable_sort_expert_major_dispatch",
    "triton_counting_scatter_expert_major_dispatch",
]
