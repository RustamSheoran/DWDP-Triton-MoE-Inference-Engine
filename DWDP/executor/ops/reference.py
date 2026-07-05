from __future__ import annotations

import torch


def gather_expert_inputs(
    flat_hidden_states: torch.Tensor,
    packed_token_indices: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather hidden states for one expert-major slice."""

    if out is None:
        return torch.index_select(flat_hidden_states, 0, packed_token_indices)
    return torch.index_select(flat_hidden_states, 0, packed_token_indices, out=out)


def apply_routing_weights(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply routing weights to expert outputs."""

    weighted = expert_outputs * routing_weights.unsqueeze(-1)
    if out is None:
        return weighted
    out.copy_(weighted)
    return out


def write_expert_outputs(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    start: int,
    end: int,
) -> None:
    """Write one expert slice into a packed output buffer."""

    destination[start:end].copy_(source)
