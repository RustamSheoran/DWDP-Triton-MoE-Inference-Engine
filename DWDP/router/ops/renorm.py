from __future__ import annotations

import torch


def renormalize_topk_weights(
    topk_probabilities: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Renormalize selected expert probabilities onto the simplex."""

    if topk_probabilities.shape[-1] == 1:
        return torch.ones_like(topk_probabilities)

    denominator = topk_probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    return topk_probabilities / denominator
