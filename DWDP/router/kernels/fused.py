from __future__ import annotations

import torch

from ..ops import renormalize_topk_weights, select_topk, stable_softmax


def reference_topk_routing(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    softmax_dtype: torch.dtype | None = None,
    probability_dtype: torch.dtype | None = None,
    topk_sorted: bool = False,
    renormalize: bool = True,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference routing path.

    This function is intentionally shaped like a future fused kernel boundary.
    A Triton or CUDA implementation can replace its internals without changing
    the public router API.
    """

    probabilities = stable_softmax(
        router_logits,
        dim=-1,
        compute_dtype=softmax_dtype,
        output_dtype=probability_dtype,
    )
    topk_probabilities, topk_indices = select_topk(
        probabilities,
        top_k=top_k,
        sorted=topk_sorted,
    )

    if renormalize:
        topk_weights = renormalize_topk_weights(topk_probabilities, eps=eps)
    else:
        topk_weights = topk_probabilities

    return probabilities, topk_indices, topk_weights
