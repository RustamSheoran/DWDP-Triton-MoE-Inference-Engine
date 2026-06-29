"""Reference dispatch tensor primitives."""

from .histogram import compute_expert_histogram
from .packing import pack_routing_weights, pack_token_indices
from .permutation import invert_permutation, stable_expert_permutation
from .prefix_sum import exclusive_cumsum
from .scatter import (
    compute_destination_positions,
    scatter_expert_ids,
    scatter_routing_weights,
    scatter_token_indices,
    scatter_token_permutation,
)

__all__ = [
    "compute_expert_histogram",
    "compute_destination_positions",
    "exclusive_cumsum",
    "invert_permutation",
    "pack_routing_weights",
    "pack_token_indices",
    "scatter_expert_ids",
    "scatter_routing_weights",
    "scatter_token_indices",
    "scatter_token_permutation",
    "stable_expert_permutation",
]
