"""Compact structure-of-arrays execution descriptor for grouped expert kernels.

``TensorList`` deliberately describes storage instead of owning it.  Activations,
weights, and output buffers keep their normal owners; :class:`ExecutorWorkspace`
owns the reusable metadata allocations.  A descriptor is valid only for the
forward call that created it, because pointer fields reference that call's
activations and output slices.

Metadata is structure-of-arrays rather than an array of Python records.  Grouped
kernels walk one field at a time (for example all input pointers), so contiguous
field arrays permit coalesced/vectorized metadata loads and avoid per-expert
Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .weights import QwenSwiGLUWeightProvider
from .workspace import ExecutorWorkspace


@dataclass(slots=True)
class TensorList:
    """A non-owning SoA descriptor for active Qwen experts.

    Every public tensor is a contiguous one-dimensional metadata array of
    ``size`` entries.  Pointer fields contain raw device addresses encoded as
    ``int64``; they never retain or copy the referenced tensor storage.  The
    workspace retains all metadata buffers and may grow their capacity, while a
    decode step merely overwrites ``[:size]``.
    """

    size: int
    capacity: int
    input_ptrs: torch.Tensor
    token_index_ptrs: torch.Tensor
    routing_weight_ptrs: torch.Tensor
    gate_weight_ptrs: torch.Tensor
    up_weight_ptrs: torch.Tensor
    down_weight_ptrs: torch.Tensor
    intermediate_ptrs: torch.Tensor
    output_ptrs: torch.Tensor
    weighted_output_ptrs: torch.Tensor
    quantization_ptrs: torch.Tensor
    expert_ids: torch.Tensor
    token_offsets: torch.Tensor
    token_counts: torch.Tensor
    m: torch.Tensor
    n: torch.Tensor
    k: torch.Tensor
    intermediate_n: torch.Tensor
    input_ld: torch.Tensor
    gate_ld: torch.Tensor
    up_ld: torch.Tensor
    down_ld: torch.Tensor
    output_ld: torch.Tensor
    dtype_codes: torch.Tensor
    workspace_indices: torch.Tensor
    execution_priorities: torch.Tensor
    stream_ids: torch.Tensor
    launch_dimensions: tuple[int, int, int]

    @classmethod
    def from_plans(
        cls,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        workspace: ExecutorWorkspace,
        provider: QwenSwiGLUWeightProvider,
        packed_outputs: torch.Tensor,
        weighted_outputs: torch.Tensor,
        intermediate: torch.Tensor,
    ) -> "TensorList":
        """Build metadata directly from finalized plans and workspace storage.

        Only scheduled experts with positive token counts are represented.  The
        single host pass is O(active experts), never scans the registry, and
        uses workspace staging arrays instead of temporary Python lists.
        """

        device = hidden_states.device
        if provider.hidden_size != hidden_states.shape[1]:
            raise ValueError("Qwen provider hidden size does not match activations")
        if packed_outputs.shape != weighted_outputs.shape:
            raise ValueError(
                "packed and weighted output buffers must have the same shape"
            )
        for name, tensor in (
            ("packed token indices", dispatch_plan.assignments.packed_token_indices),
            ("routing weights", dispatch_plan.assignments.packed_routing_weights),
            ("expert queue", execution_plan.expert_queue),
            ("expert starts", execution_plan.expert_starts),
            ("expert counts", execution_plan.expert_counts),
        ):
            if tensor.device != device:
                raise ValueError(f"TensorList {name} must share the activation device")

        # Scheduler arrays already contain only active experts in normal
        # runtime operation.  Reusable staging avoids allocating a stacked
        # schedule on every decode iteration.
        schedule = workspace.get_tensorlist_schedule(
            execution_plan.expert_queue,
            execution_plan.expert_starts,
            execution_plan.expert_counts,
            execution_plan.execution_priority,
            execution_plan.stream_assignments,
        )
        # ExecutionPlan normally contains only active experts.  Retaining the
        # positive-count guard keeps TensorList safe for externally built
        # plans without ever traversing the full expert registry.
        mask = schedule[2] > 0
        valid_schedule = schedule[:, mask]
        size = valid_schedule.shape[1]
        
        capacity = workspace.ensure_tensorlist_capacity(size, device=device)
        fields = workspace.tensorlist_device_fields
        host = workspace.tensorlist_host_fields
        assert fields is not None and host is not None

        if workspace._host_copy_event is not None:
            workspace._host_copy_event.synchronize()

        if provider.gate_up_weights.dtype != hidden_states.dtype or provider.down_weights.dtype != hidden_states.dtype:
            raise ValueError("grouped Triton execution requires matching activation and weight dtypes")
        if provider.gate_up_weights.device != device or provider.down_weights.device != device:
            raise ValueError("grouped Triton execution requires local weight storage on the activation device")

        dtype_code = _dtype_code(hidden_states.dtype)
        input_ptr = hidden_states.data_ptr()
        token_index_ptr = dispatch_plan.assignments.packed_token_indices.data_ptr()
        routing_weight_ptr = dispatch_plan.assignments.packed_routing_weights.data_ptr()

        gate_ptrs_map, up_ptrs_map, down_ptrs_map, gate_lds_map, up_lds_map, down_lds_map = workspace.get_tensorlist_provider_pointers_cpu(provider)
        
        expert_ids = valid_schedule[0]
        token_offsets = valid_schedule[1]
        token_counts = valid_schedule[2]
        priorities = valid_schedule[3]
        stream_ids = valid_schedule[4]

        host["input_ptrs"][:size] = input_ptr
        host["token_index_ptrs"][:size] = token_index_ptr
        host["routing_weight_ptrs"][:size] = routing_weight_ptr
        
        host["gate_weight_ptrs"][:size] = gate_ptrs_map[expert_ids]
        host["up_weight_ptrs"][:size] = up_ptrs_map[expert_ids]
        host["down_weight_ptrs"][:size] = down_ptrs_map[expert_ids]
        
        host["intermediate_ptrs"][:size] = intermediate.data_ptr() + token_offsets * (intermediate.stride(0) * intermediate.element_size())
        host["output_ptrs"][:size] = packed_outputs.data_ptr() + token_offsets * (packed_outputs.stride(0) * packed_outputs.element_size())
        host["weighted_output_ptrs"][:size] = weighted_outputs.data_ptr() + token_offsets * (weighted_outputs.stride(0) * weighted_outputs.element_size())
        
        host["quantization_ptrs"][:size] = 0
        host["expert_ids"][:size] = expert_ids
        host["token_offsets"][:size] = token_offsets
        host["token_counts"][:size] = token_counts
        host["m"][:size] = token_counts
        
        host["n"][:size] = packed_outputs.shape[1]
        host["k"][:size] = hidden_states.shape[1]
        host["intermediate_n"][:size] = intermediate.shape[1]
        
        host["input_ld"][:size] = hidden_states.stride(0)
        host["gate_ld"][:size] = gate_lds_map[expert_ids]
        host["up_ld"][:size] = up_lds_map[expert_ids]
        host["down_ld"][:size] = down_lds_map[expert_ids]
        host["output_ld"][:size] = packed_outputs.stride(0)
        
        host["dtype_codes"][:size] = dtype_code
        host["workspace_indices"][:size] = torch.arange(size, dtype=torch.int64)
        host["execution_priorities"][:size] = priorities
        host["stream_ids"][:size] = stream_ids
        
        max_m = int(torch.max(token_counts).item()) if size > 0 else 0

        if workspace._tensorlist_device_buffer is not None and workspace._tensorlist_host_buffer is not None:
            workspace._tensorlist_device_buffer[:, :size].copy_(
                workspace._tensorlist_host_buffer[:, :size], non_blocking=True
            )
        else:
            for name, host_field in host.items():
                fields[name][:size].copy_(host_field[:size], non_blocking=True)

        return cls(
            size=size,
            capacity=capacity,
            launch_dimensions=workspace.get_tensorlist_launch_dimensions(
                max_m, packed_outputs.shape[1], intermediate.shape[1]
            ),
            **{name: fields[name][:size] for name in _FIELD_NAMES},
        )


_FIELD_NAMES = (
    "input_ptrs",
    "token_index_ptrs",
    "routing_weight_ptrs",
    "gate_weight_ptrs",
    "up_weight_ptrs",
    "down_weight_ptrs",
    "intermediate_ptrs",
    "output_ptrs",
    "weighted_output_ptrs",
    "quantization_ptrs",
    "expert_ids",
    "token_offsets",
    "token_counts",
    "m",
    "n",
    "k",
    "intermediate_n",
    "input_ld",
    "gate_ld",
    "up_ld",
    "down_ld",
    "output_ld",
    "dtype_codes",
    "workspace_indices",
    "execution_priorities",
    "stream_ids",
)


def tensorlist_field_names() -> tuple[str, ...]:
    """Return the fixed SoA field layout used by :class:`ExecutorWorkspace`."""

    return _FIELD_NAMES


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    if dtype in _fp8_dtypes():
        return 3
    raise ValueError(
        "Triton execution requires float16, bfloat16, or a supported FP8 dtype"
    )


def _fp8_dtypes() -> tuple[torch.dtype, ...]:
    """Return FP8 dtypes exposed by this PyTorch build without version gates."""

    return tuple(
        dtype
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        )
        if isinstance((dtype := getattr(torch, name, None)), torch.dtype)
    )
