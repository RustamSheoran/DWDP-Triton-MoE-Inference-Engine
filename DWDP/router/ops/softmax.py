from __future__ import annotations

import torch

from ..utils import default_softmax_dtype


def stable_softmax(
    logits: torch.Tensor,
    dim: int = -1,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Numerically stable softmax with configurable accumulation dtype."""

    if compute_dtype is None:
        compute_dtype = default_softmax_dtype(logits.dtype)

    probabilities = torch.softmax(logits, dim=dim, dtype=compute_dtype)
    if output_dtype is not None and probabilities.dtype != output_dtype:
        probabilities = probabilities.to(output_dtype)
    return probabilities
