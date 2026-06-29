from __future__ import annotations

import torch


def exclusive_cumsum(counts: torch.Tensor) -> torch.Tensor:
    """Compute exclusive prefix offsets from counts."""

    return torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)
