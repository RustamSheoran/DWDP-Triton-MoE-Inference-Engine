from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DispatchMetadata:
    """Dispatch metadata required by later runtime stages."""

    num_tokens: int
    num_assignments: int
    num_experts: int
    top_k: int
    token_shape: tuple[int, ...]
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    token_permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    destination_positions: torch.Tensor
    stable_order: bool
    algorithm: str
