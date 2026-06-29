from __future__ import annotations

import torch


def compute_expert_histogram(
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute per-expert assignment counts."""

    histogram_input = expert_indices
    if histogram_input.dtype != torch.int64:
        histogram_input = histogram_input.to(torch.int64)
    return torch.bincount(histogram_input, minlength=num_experts)
