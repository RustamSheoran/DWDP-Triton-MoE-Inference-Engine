from __future__ import annotations

import torch

from ..workspace import SchedulerWorkspace


def build_round_robin_schedule(
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    stream_count: int,
    workspace: SchedulerWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic ascending-expert execution metadata."""

    active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten()
    active_count = active_experts.numel()

    if workspace is None:
        execution_order = torch.arange(
            active_count,
            dtype=torch.int64,
            device=expert_counts.device,
        )
        expert_queue = active_experts
        expert_starts = expert_offsets.index_select(0, active_experts)
        expert_ends = expert_offsets.index_select(0, active_experts + 1)
        active_counts = expert_counts.index_select(0, active_experts)
        execution_priority = execution_order.clone()
        stream_assignments = torch.remainder(execution_order, stream_count)
        return (
            execution_order,
            expert_queue,
            expert_starts,
            expert_ends,
            active_counts,
            execution_priority,
            stream_assignments,
        )

    (
        execution_order,
        expert_queue,
        expert_starts,
        expert_ends,
        active_counts,
        execution_priority,
        stream_assignments,
    ) = workspace.get_active_expert_buffers(active_count, device=expert_counts.device)

    torch.arange(active_count, dtype=torch.int64, device=expert_counts.device, out=execution_order)
    expert_queue.copy_(active_experts)
    torch.index_select(expert_offsets, 0, active_experts, out=expert_starts)
    torch.index_select(expert_offsets, 0, active_experts + 1, out=expert_ends)
    torch.index_select(expert_counts, 0, active_experts, out=active_counts)
    execution_priority.copy_(execution_order)
    torch.remainder(execution_order, stream_count, out=stream_assignments)
    return (
        execution_order,
        expert_queue,
        expert_starts,
        expert_ends,
        active_counts,
        execution_priority,
        stream_assignments,
    )
