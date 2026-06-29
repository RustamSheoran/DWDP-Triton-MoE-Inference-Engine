from __future__ import annotations

import torch


def select_topk(
    probabilities: torch.Tensor,
    top_k: int,
    sorted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts from router probabilities."""

    topk_values, topk_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=-1,
        largest=True,
        sorted=sorted,
    )
    return topk_values, topk_indices
