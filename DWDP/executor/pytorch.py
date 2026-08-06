from __future__ import annotations

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .base import BaseExecutor
from .config import ExecutorConfig
from .experts import ExpertRegistry
from .metadata import TimingMetadata, WorkspaceMetadata
from .ops import gather_expert_inputs
from .outputs import (
    ExecutionMetadata,
    ExecutionStatistics,
    ExecutorOutput,
    ExpertOutput,
    OutputMetadata,
)
from .registry import register_executor
from .utils import flatten_hidden_states, validate_executor_inputs
from .workspace import ExecutorWorkspace


class PyTorchExecutor(BaseExecutor):
    """Reference PyTorch backend for local expert execution."""

    def __init__(self, config: ExecutorConfig, experts: ExpertRegistry) -> None:
        super().__init__(config, experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None = None,
    ) -> ExecutorOutput:
        """Execute experts exactly in scheduler-provided order."""

        flat_hidden_states, token_shape = flatten_hidden_states(hidden_states)
        validate_executor_inputs(
            flat_hidden_states,
            dispatch_plan,
            execution_plan,
            communication_plan,
        )

        active_workspace = workspace if self.config.enable_workspace else None
        output_dtype = self.config.dtype or hidden_states.dtype
        device = hidden_states.device
        num_assignments = dispatch_plan.metadata.num_assignments
        hidden_size = hidden_states.shape[-1]
        output_size: int | None = None
        packed_outputs: torch.Tensor | None = None
        weighted_outputs: torch.Tensor | None = None

        expert_records: list[ExpertOutput] = []
        skipped_experts = 0

        schedule_rows = self._get_schedule_rows(
            execution_plan, workspace=active_workspace
        )
        for schedule_idx, (
            expert_id,
            start,
            end,
            count,
            priority,
            stream_id,
        ) in enumerate(schedule_rows):
            if count == 0:
                skipped_experts += 1
                continue
            if (
                self.config.max_tokens_per_expert is not None
                and count > self.config.max_tokens_per_expert
            ):
                raise ValueError(
                    f"Expert {expert_id} received {count} tokens, exceeding max_tokens_per_expert"
                )

            token_indices = dispatch_plan.assignments.packed_token_indices[start:end]
            routing_weights = dispatch_plan.assignments.packed_routing_weights[
                start:end
            ]
            gathered = self._gather_inputs(
                flat_hidden_states,
                token_indices,
                workspace=active_workspace,
            )
            expert_output = self._execute_expert(expert_id, gathered)
            if output_size is None:
                output_size = int(expert_output.shape[-1])
                packed_outputs, weighted_outputs = self._allocate_outputs(
                    num_assignments,
                    output_size,
                    dtype=output_dtype,
                    device=device,
                    workspace=active_workspace,
                )
            if expert_output.shape != (count, output_size):
                raise ValueError(
                    f"Expert {expert_id} returned shape {tuple(expert_output.shape)}, "
                    f"expected {(count, output_size)}"
                )

            if packed_outputs is not None:
                packed_slice = packed_outputs[start:end]
                if expert_output.dtype == output_dtype:
                    packed_slice.copy_(expert_output)
                    weight_source = packed_slice
                else:
                    packed_slice.copy_(expert_output.to(dtype=output_dtype))
                    # Preserve reference rounding: routing is applied at the
                    # expert's native precision, then written to output_dtype.
                    weight_source = expert_output
            else:
                # Packed outputs are not materialized; multiply straight from
                # the expert result. Same rounding as the branch above, which
                # also multiplies at the expert's native precision whenever the
                # dtypes differ.
                weight_source = expert_output
            # Write weighted values directly to the final packed buffer.  The
            # previous path allocated a full temporary tensor and then copied
            # it into this exact slice for every active expert.
            torch.mul(
                weight_source,
                routing_weights.unsqueeze(-1),
                out=weighted_outputs[start:end],
            )
            expert_records.append(
                ExpertOutput(
                    expert_id=expert_id,
                    start=start,
                    end=end,
                    count=count,
                    priority=priority,
                    stream_id=stream_id,
                )
            )

        if output_size is None:
            output_size = hidden_size
            packed_outputs, weighted_outputs = self._allocate_outputs(
                num_assignments,
                output_size,
                dtype=output_dtype,
                device=device,
                workspace=active_workspace,
            )
        assert weighted_outputs is not None

        output_metadata = OutputMetadata(
            packed_token_indices=dispatch_plan.assignments.packed_token_indices,
            packed_expert_ids=dispatch_plan.assignments.expert_ids,
            packed_routing_weights=dispatch_plan.assignments.packed_routing_weights,
            token_permutation=dispatch_plan.metadata.token_permutation,
            inverse_permutation=dispatch_plan.metadata.inverse_permutation,
            token_shape=token_shape,
            top_k=dispatch_plan.metadata.top_k,
        )
        execution_metadata = ExecutionMetadata(
            execution_order=execution_plan.execution_order,
            expert_queue=execution_plan.expert_queue,
            expert_starts=execution_plan.expert_starts,
            expert_ends=execution_plan.expert_ends,
            stream_assignments=execution_plan.stream_assignments,
            communication_remote_expert_ids=communication_plan.remote_expert_ids,
            communication_policy=communication_plan.planner_policy,
            scheduling_policy=execution_plan.scheduling_policy,
        )
        statistics = ExecutionStatistics(
            num_executed_experts=len(expert_records),
            num_skipped_experts=skipped_experts,
            num_input_tokens=flat_hidden_states.shape[0],
            num_assignments=num_assignments,
            hidden_size=hidden_size,
            output_size=output_size,
            backend=self.config.backend,
        )
        workspace_metadata = WorkspaceMetadata(
            used_workspace=active_workspace is not None,
            workspace_bytes=active_workspace.estimated_bytes()
            if active_workspace is not None
            else 0,
        )

        return ExecutorOutput(
            packed_expert_outputs=packed_outputs,
            weighted_expert_outputs=weighted_outputs,
            expert_outputs=tuple(expert_records),
            output_metadata=output_metadata,
            execution_metadata=execution_metadata,
            statistics=statistics,
            timing=TimingMetadata(),
            workspace=workspace_metadata,
            backend=self.config.backend,
            deterministic=self.config.deterministic,
        )

    @staticmethod
    def _get_schedule_rows(
        execution_plan: ExecutionPlan,
        *,
        workspace: ExecutorWorkspace | None,
    ) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Materialize scheduler fields without per-expert CUDA scalar reads.

        Note: this is inherently a device->host transfer, so ``PyTorchExecutor``
        cannot be CUDA-graph captured. The persistent Triton executor consumes
        the same queue tensors on device and is capturable.
        """

        if workspace is not None:
            return workspace.get_schedule_rows(
                execution_plan.expert_queue,
                execution_plan.expert_starts,
                execution_plan.expert_ends,
                execution_plan.expert_counts,
                execution_plan.execution_priority,
                execution_plan.stream_assignments,
            )
        values = (
            torch.stack(
                (
                    execution_plan.expert_queue,
                    execution_plan.expert_starts,
                    execution_plan.expert_ends,
                    execution_plan.expert_counts,
                    execution_plan.execution_priority,
                    execution_plan.stream_assignments,
                )
            )
            .cpu()
            .tolist()
        )
        return tuple(zip(*values))

    def _allocate_outputs(
        self,
        num_assignments: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        workspace: ExecutorWorkspace | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        materialize_packed = self.config.materialize_packed_outputs
        if workspace is None:
            packed = (
                torch.empty(num_assignments, output_size, dtype=dtype, device=device)
                if materialize_packed
                else None
            )
            return (
                packed,
                torch.empty(num_assignments, output_size, dtype=dtype, device=device),
            )
        return workspace.get_output_buffers(
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
            materialize_packed=materialize_packed,
        )

    def _gather_inputs(
        self,
        flat_hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        workspace: ExecutorWorkspace | None,
    ) -> torch.Tensor:
        if workspace is None:
            return gather_expert_inputs(flat_hidden_states, token_indices)
        buffer = workspace.get_gather_buffer(
            token_indices.numel(),
            flat_hidden_states.shape[-1],
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )
        if not self.config.enable_profiling:
            return gather_expert_inputs(flat_hidden_states, token_indices, out=buffer)
        with torch.autograd.profiler.record_function("dwdp.gather"):
            return gather_expert_inputs(flat_hidden_states, token_indices, out=buffer)

    def _execute_expert(
        self,
        expert_id: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        expert_ptr = self.communication_engine.getResidentPointer(expert_id)
        if self.config.enable_profiling:
            with torch.autograd.profiler.record_function("dwdp.expert_gemms"):
                expert_outputs = expert_ptr.module(hidden_states)
        else:
            expert_outputs = expert_ptr.module(hidden_states)
        if expert_outputs.ndim != 2:
            raise ValueError("Expert outputs must be rank-2 [tokens, output_dim]")
        return expert_outputs


register_executor("pytorch", PyTorchExecutor)
