from __future__ import annotations

import torch

from DWDP.dispatcher.plan import DispatchPlan


def validate_dispatch_plan(dispatch_plan: DispatchPlan) -> None:
    """Validate the dispatch contract consumed by the scheduler."""

    metadata = dispatch_plan.metadata
    if metadata.expert_counts.ndim != 1:
        raise ValueError("expert_counts must be 1D")
    if metadata.expert_offsets.ndim != 1:
        raise ValueError("expert_offsets must be 1D")
    if metadata.expert_counts.numel() != metadata.num_experts:
        raise ValueError("expert_counts length must match num_experts")
    if metadata.expert_offsets.numel() != metadata.num_experts + 1:
        raise ValueError("expert_offsets length must be num_experts + 1")
    if metadata.expert_counts.dtype != torch.int64:
        raise ValueError("expert_counts must be int64")
    if metadata.expert_offsets.dtype != torch.int64:
        raise ValueError("expert_offsets must be int64")
    if metadata.expert_offsets.device != metadata.expert_counts.device:
        raise ValueError("expert_offsets and expert_counts must be on the same device")


def make_execution_batches(
    expert_queue: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_ends: torch.Tensor,
    expert_counts: torch.Tensor,
    execution_priority: torch.Tensor,
    stream_assignments: torch.Tensor,
) -> tuple:
    """Materialize Python execution descriptors for executor-facing metadata."""

    from .execution import ExecutionBatch

    q_list = expert_queue.tolist()
    s_list = expert_starts.tolist()
    e_list = expert_ends.tolist()
    c_list = expert_counts.tolist()
    p_list = execution_priority.tolist()
    st_list = stream_assignments.tolist()

    batches = [
        ExecutionBatch(
            expert_id=int(q_list[idx]),
            start=int(s_list[idx]),
            end=int(e_list[idx]),
            count=int(c_list[idx]),
            priority=int(p_list[idx]),
            stream_id=int(st_list[idx]),
        )
        for idx in range(len(q_list))
    ]
    return tuple(batches)


def estimate_tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Estimate tensor storage size in bytes."""

    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
