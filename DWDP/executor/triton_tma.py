"""Hopper TMA FP8 executor backend with fine-grained micro-scaling.

This is the ``triton_fp8_tma`` backend. It runs the same grouped-GEMM shape as
the persistent engine but drives each expert's projections through the TMA
kernel in ``kernels/fp8_tma.py``, using per-128-element scale factors instead
of a single scalar per expert (Project 1 "Weight Quantization" row).

Weights are quantized once, lazily, on the first CUDA forward and cached
per expert. Activations are quantized per forward, since their dynamic range
changes with the batch.

On any device or Triton build without TMA support the backend falls back to
the standard persistent Triton path, so selecting it is never fatal.
"""

from __future__ import annotations

import logging

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .config import ExecutorConfig
from .experts import ExpertRegistry
from .fp8 import (
    quantize_activations_blockwise,
    quantize_weights_blockwise,
)
from .kernels.fp8_tma import fp8_tma_microscaled_gemm, hopper_tma_supported
from .metadata import TimingMetadata, WorkspaceMetadata
from .outputs import (
    ExecutionMetadata,
    ExecutionStatistics,
    ExecutorOutput,
    ExpertOutput,
    OutputMetadata,
)
from .registry import register_executor
from .triton import TritonExpertExecutor
from .utils import flatten_hidden_states, validate_executor_inputs
from .weights import ExpertWeightProvider
from .workspace import ExecutorWorkspace

logger = logging.getLogger(__name__)


class TritonTMAFP8Executor(TritonExpertExecutor):
    """Grouped MoE execution through Hopper TMA FP8 GEMMs."""

    def __init__(
        self,
        config: ExecutorConfig,
        experts: ExpertRegistry,
        weight_provider: ExpertWeightProvider | None = None,
    ) -> None:
        super().__init__(config, experts, weight_provider)
        self._fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        # expert_id -> (gate_q, gate_s, up_q, up_s, down_q, down_s)
        self._quantized: dict[int, tuple[torch.Tensor, ...]] = {}

    def _quantize_expert(self, expert_id: int) -> tuple[torch.Tensor, ...]:
        """Quantize one expert's three projections, caching the result."""

        cached = self._quantized.get(expert_id)
        if cached is not None:
            return cached

        dtype = self._fp8_dtype
        gate, up = self.weight_provider.gate_up_weights.for_expert(expert_id)
        down = self.weight_provider.down_weights.for_expert(expert_id)

        with torch.no_grad():
            entry = (
                *quantize_weights_blockwise(gate, dtype),
                *quantize_weights_blockwise(up, dtype),
                *quantize_weights_blockwise(down, dtype),
            )
        self._quantized[expert_id] = entry
        return entry

    def _tma_active(self, hidden_states: torch.Tensor) -> bool:
        return (
            self._fp8_dtype is not None
            and hidden_states.is_cuda
            and hopper_tma_supported(hidden_states.device)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None = None,
    ) -> ExecutorOutput:
        """Execute experts through TMA FP8 GEMMs, or fall back when unsupported."""

        if not self._tma_active(hidden_states):
            output = super().forward(
                hidden_states,
                dispatch_plan,
                execution_plan,
                communication_plan,
                workspace=workspace,
            )
            return output

        return self._forward_tma(
            hidden_states, dispatch_plan, execution_plan, communication_plan, workspace
        )

    def _forward_tma(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None,
    ) -> ExecutorOutput:
        """TMA path: weights quantized once per expert, activations per forward.

        Emits packed (assignment-major) rows exactly like the persistent
        backend, so the Merger stage is unchanged.
        """

        flat_hidden_states, token_shape = flatten_hidden_states(hidden_states)
        validate_executor_inputs(
            flat_hidden_states, dispatch_plan, execution_plan, communication_plan
        )
        if self.weight_provider.has_bias:
            raise ValueError("TMA FP8 execution does not support biased projections")

        active_workspace = workspace or ExecutorWorkspace()
        assignments = dispatch_plan.metadata.num_assignments
        output_dtype = self.config.dtype or hidden_states.dtype

        packed_outputs, weighted_outputs = active_workspace.get_output_buffers(
            assignments,
            self.weight_provider.hidden_size,
            dtype=output_dtype,
            device=hidden_states.device,
        )
        packed_outputs.zero_()
        weighted_outputs.zero_()

        # One scale per (token, 128-channel block); recomputed each forward
        # because activation range moves with the batch.
        fp8_activations, activation_scales = quantize_activations_blockwise(
            flat_hidden_states, self._fp8_dtype
        )

        packed_token_indices = dispatch_plan.assignments.packed_token_indices
        packed_routing_weights = dispatch_plan.assignments.packed_routing_weights

        # Iterate the plan tensors rather than execution_plan.batches: the
        # Python batch descriptors are only materialized at FULL scheduler
        # metadata, and the adapter runs the scheduler at MINIMAL.
        queue = execution_plan.expert_queue.tolist()
        starts = execution_plan.expert_starts.tolist()
        ends = execution_plan.expert_ends.tolist()
        counts = execution_plan.expert_counts.tolist()
        priorities = execution_plan.execution_priority.tolist()
        streams = execution_plan.stream_assignments.tolist()

        records: list[ExpertOutput] = []
        executed = 0
        for index, expert_id in enumerate(queue):
            count = int(counts[index])
            if count <= 0:
                continue
            executed += 1
            expert_id = int(expert_id)
            start, end = int(starts[index]), int(ends[index])

            gate_q, gate_s, up_q, up_s, down_q, down_s = self._quantize_expert(
                expert_id
            )

            rows = packed_token_indices[start:end]
            batch_activations = fp8_activations.index_select(0, rows)
            batch_scales = activation_scales.index_select(0, rows)

            gate_out = fp8_tma_microscaled_gemm(
                batch_activations, gate_q, batch_scales, gate_s, out_dtype=output_dtype
            )
            up_out = fp8_tma_microscaled_gemm(
                batch_activations, up_q, batch_scales, up_s, out_dtype=output_dtype
            )
            intermediate = gate_out * torch.sigmoid(gate_out) * up_out

            # The SwiGLU result is a fresh activation, so it needs its own
            # block scales before the down projection.
            intermediate_q, intermediate_s = quantize_activations_blockwise(
                intermediate, self._fp8_dtype
            )
            expert_out = fp8_tma_microscaled_gemm(
                intermediate_q, down_q, intermediate_s, down_s, out_dtype=output_dtype
            )

            weights = packed_routing_weights[start:end]
            packed_outputs[start:end] = expert_out
            weighted_outputs[start:end] = expert_out * weights.unsqueeze(-1).to(
                expert_out.dtype
            )

            records.append(
                ExpertOutput(
                    expert_id=expert_id,
                    start=start,
                    end=end,
                    count=count,
                    priority=int(priorities[index]),
                    stream_id=int(streams[index]),
                )
            )

        return ExecutorOutput(
            packed_expert_outputs=packed_outputs,
            weighted_expert_outputs=weighted_outputs,
            expert_outputs=tuple(records),
            output_metadata=OutputMetadata(
                packed_token_indices=packed_token_indices,
                packed_expert_ids=dispatch_plan.assignments.expert_ids,
                packed_routing_weights=packed_routing_weights,
                token_permutation=dispatch_plan.metadata.token_permutation,
                inverse_permutation=dispatch_plan.metadata.inverse_permutation,
                token_shape=token_shape,
                top_k=dispatch_plan.metadata.top_k,
            ),
            execution_metadata=ExecutionMetadata(
                execution_order=execution_plan.execution_order,
                expert_queue=execution_plan.expert_queue,
                expert_starts=execution_plan.expert_starts,
                expert_ends=execution_plan.expert_ends,
                stream_assignments=execution_plan.stream_assignments,
                communication_remote_expert_ids=communication_plan.remote_expert_ids,
                communication_policy=communication_plan.planner_policy,
                scheduling_policy=execution_plan.scheduling_policy,
            ),
            statistics=ExecutionStatistics(
                num_executed_experts=executed,
                num_skipped_experts=execution_plan.expert_queue.numel() - executed,
                num_input_tokens=flat_hidden_states.shape[0],
                num_assignments=assignments,
                hidden_size=flat_hidden_states.shape[1],
                output_size=self.weight_provider.hidden_size,
                backend="triton_fp8_tma",
            ),
            timing=TimingMetadata(),
            workspace=WorkspaceMetadata(
                used_workspace=workspace is not None,
                workspace_bytes=0,
            ),
            backend="triton_fp8_tma",
            deterministic=self.config.deterministic,
        )


register_executor("triton_fp8_tma", TritonTMAFP8Executor)
