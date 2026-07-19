from __future__ import annotations

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .base import BaseExecutor
from .config import ExecutorConfig
from .experts import ExpertBatch, ExpertExecutionContext, ExpertRegistry
from .metadata import TimingMetadata, WorkspaceMetadata
from .ops import apply_routing_weights, gather_expert_inputs, write_expert_outputs
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

        for schedule_idx in range(execution_plan.expert_queue.numel()):
            expert_id = int(execution_plan.expert_queue[schedule_idx].item())
            start = int(execution_plan.expert_starts[schedule_idx].item())
            end = int(execution_plan.expert_ends[schedule_idx].item())
            count = int(execution_plan.expert_counts[schedule_idx].item())
            priority = int(execution_plan.execution_priority[schedule_idx].item())
            stream_id = int(execution_plan.stream_assignments[schedule_idx].item())

            if count == 0:
                skipped_experts += 1
                continue
            if self.config.max_tokens_per_expert is not None and count > self.config.max_tokens_per_expert:
                raise ValueError(
                    f"Expert {expert_id} received {count} tokens, exceeding max_tokens_per_expert"
                )

            token_indices = dispatch_plan.assignments.packed_token_indices[start:end]
            routing_weights = dispatch_plan.assignments.packed_routing_weights[start:end]
            gathered = self._gather_inputs(
                flat_hidden_states,
                token_indices,
                workspace=active_workspace,
            )
            batch = ExpertBatch(
                expert_id=expert_id,
                start=start,
                end=end,
                token_indices=token_indices,
                routing_weights=routing_weights,
                hidden_states=gathered,
            )
            context = ExpertExecutionContext(
                expert_id=expert_id,
                priority=priority,
                stream_id=stream_id,
                deterministic=self.config.deterministic,
            )
            expert_output, weighted_output = self._execute_expert(batch, context)
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

            expert_output = expert_output.to(dtype=output_dtype)
            weighted_output = weighted_output.to(dtype=output_dtype)
            write_expert_outputs(packed_outputs, expert_output, start=start, end=end)
            write_expert_outputs(weighted_outputs, weighted_output, start=start, end=end)
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
        assert packed_outputs is not None
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
            workspace_bytes=active_workspace.estimated_bytes() if active_workspace is not None else 0,
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

    def _allocate_outputs(
        self,
        num_assignments: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        workspace: ExecutorWorkspace | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if workspace is None:
            return (
                torch.empty(num_assignments, output_size, dtype=dtype, device=device),
                torch.empty(num_assignments, output_size, dtype=dtype, device=device),
            )
        return workspace.get_output_buffers(
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )

    def _gather_inputs(
        self,
        flat_hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        workspace: ExecutorWorkspace | None,
    ) -> torch.Tensor:
        with torch.autograd.profiler.record_function("dwdp.gather"):
            if workspace is None:
                return gather_expert_inputs(flat_hidden_states, token_indices)
            buffer = workspace.get_gather_buffer(
                token_indices.numel(),
                flat_hidden_states.shape[-1],
                dtype=flat_hidden_states.dtype,
                device=flat_hidden_states.device,
            )
            return gather_expert_inputs(flat_hidden_states, token_indices, out=buffer)

    def _execute_expert(
        self,
        batch: ExpertBatch,
        context: ExpertExecutionContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        expert = self.experts.get(batch.expert_id)
        with torch.autograd.profiler.record_function("dwdp.expert_gemms"):
            expert_outputs = expert(batch.hidden_states)
        if expert_outputs.ndim != 2:
            raise ValueError("Expert outputs must be rank-2 [tokens, output_dim]")
        weighted_outputs = apply_routing_weights(expert_outputs, batch.routing_weights)
        return expert_outputs, weighted_outputs


register_executor("pytorch", PyTorchExecutor)
