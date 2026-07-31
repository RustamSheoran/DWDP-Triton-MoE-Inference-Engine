from __future__ import annotations

import torch


def exclusive_cumsum(counts: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute exclusive prefix offsets from counts."""

    if out is None:
        out = torch.zeros(counts.numel() + 1, dtype=counts.dtype, device=counts.device)
    else:
        out.zero_()

    out[1:] = torch.cumsum(counts, dim=0)
    return out
