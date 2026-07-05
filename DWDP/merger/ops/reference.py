from __future__ import annotations

import torch


def restore_token_major_assignments(
    expert_major_outputs: torch.Tensor,
    inverse_permutation: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restore packed expert-major outputs to token-major assignment order."""

    if out is None:
        return torch.index_select(expert_major_outputs, 0, inverse_permutation)
    return torch.index_select(expert_major_outputs, 0, inverse_permutation, out=out)


def reduce_topk_assignments(
    token_major_assignments: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce token-major top-k expert contributions into one output per token."""

    output_size = token_major_assignments.shape[-1]
    view = token_major_assignments.reshape(num_tokens, top_k, output_size)
    reduced = view.sum(dim=1)
    if out is None:
        return reduced
    out.copy_(reduced)
    return out
