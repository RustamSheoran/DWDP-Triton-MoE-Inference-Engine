from __future__ import annotations

import torch


def reference_merge(
    expert_major_outputs: torch.Tensor,
    packed_token_indices: torch.Tensor,
    *,
    num_tokens: int,
    merged_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate expert-major rows directly into token-major output."""

    output_size = expert_major_outputs.shape[-1]
    if merged_out is None:
        merged = torch.zeros(
            num_tokens,
            output_size,
            dtype=expert_major_outputs.dtype,
            device=expert_major_outputs.device,
        )
    else:
        merged = merged_out
        merged.zero_()
    merged.index_add_(0, packed_token_indices, expert_major_outputs)
    return merged
