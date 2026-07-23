"""Kernel replacement boundaries for future optimized expert execution."""

from .grouped_matmul import (
    TRITON_AVAILABLE,
    grouped_matmul,
    grouped_matmul_from_dispatch,
    materialize_expert_major_weights,
    reference_grouped_matmul,
)
from .reference import reference_execute_expert

__all__ = [
    "TRITON_AVAILABLE",
    "grouped_matmul",
    "grouped_matmul_from_dispatch",
    "materialize_expert_major_weights",
    "reference_execute_expert",
    "reference_grouped_matmul",
]
