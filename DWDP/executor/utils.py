from __future__ import annotations

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan


def flatten_hidden_states(hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten token-major hidden states for expert execution."""

    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least 2 dimensions")
    return hidden_states.reshape(-1, hidden_states.shape[-1]), tuple(hidden_states.shape[:-1])


def validate_executor_inputs(
    flat_hidden_states: torch.Tensor,
    dispatch_plan: DispatchPlan,
    execution_plan: ExecutionPlan,
    communication_plan: CommunicationPlan,
) -> None:
    """Validate finalized planning contracts before execution."""

    metadata = dispatch_plan.metadata
    assignments = dispatch_plan.assignments
    num_assignments = metadata.num_assignments

    if assignments.packed_token_indices.numel() != num_assignments:
        raise ValueError("packed_token_indices length must match num_assignments")
    if assignments.packed_routing_weights.numel() != num_assignments:
        raise ValueError("packed_routing_weights length must match num_assignments")
    if assignments.expert_ids.numel() != num_assignments:
        raise ValueError("expert_ids length must match num_assignments")
    if metadata.token_permutation.numel() != num_assignments:
        raise ValueError("token_permutation length must match num_assignments")
    if metadata.inverse_permutation.numel() != num_assignments:
        raise ValueError("inverse_permutation length must match num_assignments")
    if flat_hidden_states.shape[0] < metadata.num_tokens:
        raise ValueError("hidden_states do not contain enough flattened tokens")

    active_count = execution_plan.expert_queue.numel()
    for tensor in (
        execution_plan.execution_order,
        execution_plan.expert_starts,
        execution_plan.expert_ends,
        execution_plan.expert_counts,
        execution_plan.execution_priority,
        execution_plan.stream_assignments,
    ):
        if tensor.numel() != active_count:
            raise ValueError("ExecutionPlan tensors must have matching active expert length")
        if tensor.dtype != torch.int64:
            raise ValueError("ExecutionPlan tensors must be int64")

    if communication_plan.remote_expert_ids.numel() > 0:
        raise NotImplementedError("PyTorchExecutor reference backend only supports local experts")


def estimate_tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Estimate tensor storage size in bytes."""

    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
