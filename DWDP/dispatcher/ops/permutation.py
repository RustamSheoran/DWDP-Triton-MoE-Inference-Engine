from __future__ import annotations

import torch


def stable_expert_permutation(
    expert_indices: torch.Tensor,
    *,
    stable_order: bool = True,
    expert_ids_out: torch.Tensor | None = None,
    permutation_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an expert-major permutation while preserving token-major order."""

    if expert_ids_out is not None and permutation_out is not None:
        torch.sort(
            expert_indices,
            stable=stable_order,
            out=(expert_ids_out, permutation_out),
        )
        return expert_ids_out, permutation_out

    packed_expert_ids, token_permutation = torch.sort(
        expert_indices,
        stable=stable_order,
    )
    return packed_expert_ids, token_permutation


def invert_permutation(
    token_permutation: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert a permutation vector."""

    if out is None:
        out = torch.empty_like(token_permutation)
    positions = torch.arange(
        token_permutation.numel(),
        device=token_permutation.device,
        dtype=token_permutation.dtype,
    )
    out.scatter_(0, token_permutation, positions)
    return out
