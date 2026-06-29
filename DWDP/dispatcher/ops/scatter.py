from __future__ import annotations

import torch


def compute_destination_positions(
    flat_expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute expert-major destination slots with an O(N) counting pass.

    The reference implementation performs the keyed prefix-count on the host to
    preserve deterministic behavior without relying on a custom GPU kernel.
    Future Triton/CUDA implementations should replace this function with a
    device-resident keyed scan.
    """

    device = flat_expert_indices.device
    expert_indices_cpu = flat_expert_indices.to(device="cpu", dtype=torch.int64)
    expert_offsets_cpu = expert_offsets.to(device="cpu", dtype=torch.int64)

    cursors = expert_offsets_cpu[:-1].tolist()
    expert_ids = expert_indices_cpu.tolist()
    destination_positions = [0] * len(expert_ids)
    for assignment_idx, expert_id in enumerate(expert_ids):
        destination = cursors[expert_id]
        destination_positions[assignment_idx] = destination
        cursors[expert_id] = destination + 1

    return torch.tensor(
        destination_positions,
        dtype=torch.int64,
        device=device,
    )


def scatter_expert_ids(
    flat_expert_indices: torch.Tensor,
    destination_positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter expert ids into expert-major order."""

    if out is None:
        out = torch.empty_like(flat_expert_indices)
    out.scatter_(0, destination_positions, flat_expert_indices)
    return out


def scatter_token_permutation(
    destination_positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter source assignment positions into expert-major order."""

    if out is None:
        out = torch.empty_like(destination_positions)
    source_positions = torch.arange(
        destination_positions.numel(),
        device=destination_positions.device,
        dtype=destination_positions.dtype,
    )
    out.scatter_(0, destination_positions, source_positions)
    return out


def scatter_token_indices(
    destination_positions: torch.Tensor,
    top_k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter token indices into expert-major order."""

    if out is None:
        out = torch.empty_like(destination_positions)
    source_positions = torch.arange(
        destination_positions.numel(),
        device=destination_positions.device,
        dtype=destination_positions.dtype,
    )
    token_indices = torch.floor_divide(source_positions, top_k)
    out.scatter_(0, destination_positions, token_indices)
    return out


def scatter_routing_weights(
    flat_routing_weights: torch.Tensor,
    destination_positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter routing weights into expert-major order."""

    if out is None:
        out = torch.empty_like(flat_routing_weights)
    out.scatter_(0, destination_positions, flat_routing_weights)
    return out
