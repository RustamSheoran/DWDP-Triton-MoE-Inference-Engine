from __future__ import annotations

import torch

from ..ops import (
    compute_destination_positions,
    compute_expert_histogram,
    exclusive_cumsum,
    invert_permutation,
    pack_routing_weights,
    pack_token_indices,
    scatter_expert_ids,
    scatter_routing_weights,
    scatter_token_indices,
    scatter_token_permutation,
    stable_expert_permutation,
)
from ..workspace import DispatchWorkspace


def _resolve_expert_layout(
    *,
    num_assignments: int,
    num_experts: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    workspace: DispatchWorkspace | None,
    router_counts: torch.Tensor | None,
    router_offsets: torch.Tensor | None,
    flat_expert_indices: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
]:
    """Resolve reusable buffers and expert histogram metadata."""

    assignment_buffers = None
    expert_buffers = None
    if workspace is not None:
        assignment_buffers = workspace.get_assignment_buffers(
            num_assignments,
            weight_dtype=weight_dtype,
            device=device,
        )
        expert_buffers = workspace.get_expert_buffers(
            num_experts,
            device=device,
        )

    if router_counts is None:
        computed_counts = compute_expert_histogram(flat_expert_indices, num_experts)
        if expert_buffers is not None:
            expert_counts = expert_buffers[0]
            expert_counts.copy_(computed_counts)
        else:
            expert_counts = computed_counts
    else:
        expert_counts = router_counts

    if router_offsets is None:
        computed_offsets = exclusive_cumsum(expert_counts)
        if expert_buffers is not None:
            expert_offsets = expert_buffers[1]
            expert_offsets.copy_(computed_offsets)
        else:
            expert_offsets = computed_offsets
    else:
        expert_offsets = router_offsets

    return assignment_buffers, expert_buffers, expert_counts, expert_offsets


def stable_sort_expert_major_dispatch(
    flat_expert_indices: torch.Tensor,
    flat_routing_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    stable_order: bool = True,
    workspace: DispatchWorkspace | None = None,
    router_counts: torch.Tensor | None = None,
    router_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable-sort reference dispatch path.

    This preserves the original baseline implementation for benchmarking and
    correctness comparison.
    """

    assignment_buffers, _, expert_counts, expert_offsets = _resolve_expert_layout(
        num_assignments=flat_expert_indices.numel(),
        num_experts=num_experts,
        device=flat_expert_indices.device,
        weight_dtype=flat_routing_weights.dtype,
        workspace=workspace,
        router_counts=router_counts,
        router_offsets=router_offsets,
        flat_expert_indices=flat_expert_indices,
    )

    if assignment_buffers is None:
        packed_expert_ids, token_permutation = stable_expert_permutation(
            flat_expert_indices,
            stable_order=stable_order,
        )
        inverse_permutation = invert_permutation(token_permutation)
        packed_token_indices = pack_token_indices(token_permutation, top_k=top_k)
        packed_routing_weights = pack_routing_weights(
            flat_routing_weights,
            token_permutation,
        )
    else:
        (
            token_permutation,
            inverse_permutation,
            packed_expert_ids,
            packed_token_indices,
            packed_routing_weights,
        ) = assignment_buffers
        packed_expert_ids, token_permutation = stable_expert_permutation(
            flat_expert_indices,
            stable_order=stable_order,
            expert_ids_out=packed_expert_ids,
            permutation_out=token_permutation,
        )
        inverse_permutation = invert_permutation(
            token_permutation,
            out=inverse_permutation,
        )
        packed_token_indices = pack_token_indices(
            token_permutation,
            top_k=top_k,
            out=packed_token_indices,
        )
        packed_routing_weights = pack_routing_weights(
            flat_routing_weights,
            token_permutation,
            out=packed_routing_weights,
        )

    return (
        expert_counts,
        expert_offsets,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
    )


def counting_scatter_expert_major_dispatch(
    flat_expert_indices: torch.Tensor,
    flat_routing_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    stable_order: bool = True,
    workspace: DispatchWorkspace | None = None,
    router_counts: torch.Tensor | None = None,
    router_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """O(N) histogram + prefix-sum + scatter reference dispatch path.

    The current reference implementation computes deterministic destination
    positions with a host-side counting pass. Future Triton/CUDA kernels should
    replace that keyed scan with a device-resident implementation.
    """

    if not stable_order:
        raise ValueError("counting_scatter requires stable_order=True in the reference path")

    assignment_buffers, _, expert_counts, expert_offsets = _resolve_expert_layout(
        num_assignments=flat_expert_indices.numel(),
        num_experts=num_experts,
        device=flat_expert_indices.device,
        weight_dtype=flat_routing_weights.dtype,
        workspace=workspace,
        router_counts=router_counts,
        router_offsets=router_offsets,
        flat_expert_indices=flat_expert_indices,
    )

    if assignment_buffers is None:
        token_permutation = None
        inverse_permutation = None
        packed_expert_ids = None
        packed_token_indices = None
        packed_routing_weights = None
    else:
        (
            token_permutation,
            inverse_permutation,
            packed_expert_ids,
            packed_token_indices,
            packed_routing_weights,
        ) = assignment_buffers

    destination_positions = compute_destination_positions(
        flat_expert_indices,
        expert_offsets,
    )

    if inverse_permutation is None:
        inverse_permutation = destination_positions
    else:
        inverse_permutation.copy_(destination_positions)
    token_permutation = scatter_token_permutation(
        destination_positions,
        out=token_permutation,
    )
    packed_expert_ids = scatter_expert_ids(
        flat_expert_indices,
        destination_positions,
        out=packed_expert_ids,
    )
    packed_token_indices = scatter_token_indices(
        destination_positions,
        top_k=top_k,
        out=packed_token_indices,
    )
    packed_routing_weights = scatter_routing_weights(
        flat_routing_weights,
        destination_positions,
        out=packed_routing_weights,
    )

    return (
        expert_counts,
        expert_offsets,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
    )


def reference_expert_major_dispatch(
    flat_expert_indices: torch.Tensor,
    flat_routing_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    algorithm: str = "counting_scatter",
    stable_order: bool = True,
    workspace: DispatchWorkspace | None = None,
    router_counts: torch.Tensor | None = None,
    router_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch planning entry point with selectable reference algorithms."""

    if algorithm == "counting_scatter":
        return counting_scatter_expert_major_dispatch(
            flat_expert_indices,
            flat_routing_weights,
            num_experts=num_experts,
            top_k=top_k,
            stable_order=stable_order,
            workspace=workspace,
            router_counts=router_counts,
            router_offsets=router_offsets,
        )
    if algorithm == "stable_sort":
        return stable_sort_expert_major_dispatch(
            flat_expert_indices,
            flat_routing_weights,
            num_experts=num_experts,
            top_k=top_k,
            stable_order=stable_order,
            workspace=workspace,
            router_counts=router_counts,
            router_offsets=router_offsets,
        )
    raise ValueError(f"Unknown dispatcher algorithm '{algorithm}'")
