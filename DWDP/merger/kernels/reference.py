from __future__ import annotations

import torch

from ..ops import reduce_topk_assignments, restore_token_major_assignments


def reference_merge(
    expert_major_outputs: torch.Tensor,
    inverse_permutation: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    assignment_out: torch.Tensor | None = None,
    merged_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference merge boundary for future fused reductions."""

    token_major = restore_token_major_assignments(
        expert_major_outputs,
        inverse_permutation,
        out=assignment_out,
    )
    merged = reduce_topk_assignments(
        token_major,
        num_tokens=num_tokens,
        top_k=top_k,
        out=merged_out,
    )
    return token_major, merged
