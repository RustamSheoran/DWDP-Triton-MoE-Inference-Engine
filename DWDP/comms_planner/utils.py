from __future__ import annotations

import torch

from DWDP.scheduler.execution import ExecutionPlan


def validate_execution_plan(execution_plan: ExecutionPlan) -> None:
    """Validate the scheduler contract consumed by the Comms Planner."""

    tensors = (
        execution_plan.execution_order,
        execution_plan.expert_queue,
        execution_plan.expert_starts,
        execution_plan.expert_ends,
        execution_plan.expert_counts,
        execution_plan.execution_priority,
        execution_plan.stream_assignments,
    )
    active_count = execution_plan.expert_queue.numel()
    for tensor in tensors:
        if tensor.ndim != 1:
            raise ValueError("ExecutionPlan scheduling tensors must be 1D")
        if tensor.numel() != active_count:
            raise ValueError("ExecutionPlan scheduling tensors must have matching lengths")
        if tensor.dtype != torch.int64:
            raise ValueError("ExecutionPlan scheduling tensors must be int64")


def empty_int64(device: torch.device) -> torch.Tensor:
    """Allocate an empty int64 tensor."""

    return torch.empty(0, dtype=torch.int64, device=device)


def empty_float32(device: torch.device) -> torch.Tensor:
    """Allocate an empty float32 tensor."""

    return torch.empty(0, dtype=torch.float32, device=device)


def estimate_tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Estimate tensor storage size in bytes."""

    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
