from __future__ import annotations

import torch


def pack_token_indices(
    token_permutation: torch.Tensor,
    top_k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert flattened assignment permutation into token indices."""

    if out is None:
        return token_permutation // top_k
    torch.div(token_permutation, top_k, rounding_mode="floor", out=out)
    return out


def pack_routing_weights(
    flat_routing_weights: torch.Tensor,
    token_permutation: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack routing weights into expert-major order."""

    if out is None:
        return torch.index_select(flat_routing_weights, 0, token_permutation)
    return torch.index_select(flat_routing_weights, 0, token_permutation, out=out)
