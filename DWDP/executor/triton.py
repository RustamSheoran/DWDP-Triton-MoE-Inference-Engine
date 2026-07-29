"""Grouped Triton execution backend for storage-preserving Qwen SwiGLU experts."""

from __future__ import annotations

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .config import ExecutorConfig
from .experts import ExpertRegistry
from .extractors import extract_qwen_swiglu_weight_provider
from .kernels.dwdp_grouped import TRITON_AVAILABLE, grouped_qwen_swiglu
from .metadata import TimingMetadata, WorkspaceMetadata
from .outputs import ExecutionMetadata, ExecutionStatistics, ExecutorOutput, ExpertOutput, OutputMetadata
from .pytorch import PyTorchExecutor
from .registry import register_executor
from .tensor_list import TensorList
from .utils import flatten_hidden_states, validate_executor_inputs
from .weights import ExpertWeightProvider, QwenSwiGLUWeightProvider
from .workspace import ExecutorWorkspace


class TritonExpertExecutor(PyTorchExecutor):
    """Execute Qwen experts through a grouped, pointer-described Triton path.

    The public Executor contract and runtime order remain unchanged.  On CUDA,
    the only object passed to the grouped backend is a :class:`TensorList`;
    grouped kernels neither import nor inspect pipeline plans.  CPU and
    Triton-less installations retain the reference path so the runtime remains
    usable in development environments without a GPU.
    """

    def __init__(
        self,
        config: ExecutorConfig,
        experts: ExpertRegistry,
        weight_provider: ExpertWeightProvider | None = None,
    ) -> None:
        super().__init__(config, experts)
        provider = weight_provider or extract_qwen_swiglu_weight_provider(experts)
        if not isinstance(provider, QwenSwiGLUWeightProvider):
            raise ValueError("grouped Triton execution requires a QwenSwiGLUWeightProvider")
        self.weight_provider = provider

    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None = None,
    ) -> ExecutorOutput:
        """Run the grouped CUDA path, falling back only where CUDA is absent."""

        if not hidden_states.is_cuda or not TRITON_AVAILABLE:
            output = super().forward(
                hidden_states, dispatch_plan, execution_plan, communication_plan, workspace=workspace
            )
            output.backend = "triton_reference_fallback"
            output.statistics.backend = "triton_reference_fallback"
            return output

        flat_hidden_states, token_shape = flatten_hidden_states(hidden_states)
        validate_executor_inputs(flat_hidden_states, dispatch_plan, execution_plan, communication_plan)
        if self.weight_provider.has_bias:
            raise ValueError("grouped Triton execution does not support biased Qwen projections")
        output_dtype = self.config.dtype or hidden_states.dtype
        if output_dtype != hidden_states.dtype:
            raise ValueError("grouped Triton execution requires output dtype to match activation dtype")
        if flat_hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("grouped Triton execution supports float16 and bfloat16 activations")
        if self.config.max_tokens_per_expert is not None:
            counts = execution_plan.expert_counts
            if bool((counts > self.config.max_tokens_per_expert).any().item()):
                raise ValueError("an expert received more than max_tokens_per_expert tokens")

        # A workspace is mandatory for descriptor lifetime and allocation
        # ownership.  A caller that omits it gets a private, forward-scoped one.
        active_workspace = workspace or ExecutorWorkspace()
        assignments = dispatch_plan.metadata.num_assignments
        packed_outputs, weighted_outputs = active_workspace.get_output_buffers(
            assignments,
            self.weight_provider.hidden_size,
            dtype=output_dtype,
            device=hidden_states.device,
        )
        intermediate = active_workspace.get_intermediate_buffer(
            assignments,
            self.weight_provider.intermediate_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        tensors = TensorList.from_plans(
            flat_hidden_states,
            dispatch_plan,
            execution_plan,
            active_workspace,
            self.weight_provider,
            packed_outputs,
            weighted_outputs,
            intermediate,
        )
        grouped_qwen_swiglu(tensors)

        host = active_workspace.tensorlist_host_fields
        assert host is not None
        records = tuple(
            ExpertOutput(
                expert_id=int(host["expert_ids"][index]),
                start=int(host["token_offsets"][index]),
                end=int(host["token_offsets"][index] + host["token_counts"][index]),
                count=int(host["token_counts"][index]),
                priority=int(host["execution_priorities"][index]),
                stream_id=int(host["stream_ids"][index]),
            )
            for index in range(tensors.size)
        )
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
            num_executed_experts=tensors.size,
            num_skipped_experts=execution_plan.expert_queue.numel() - tensors.size,
            num_input_tokens=flat_hidden_states.shape[0],
            num_assignments=assignments,
            hidden_size=flat_hidden_states.shape[1],
            output_size=self.weight_provider.hidden_size,
            backend=self.config.backend,
        )
        return ExecutorOutput(
            packed_expert_outputs=packed_outputs,
            weighted_expert_outputs=weighted_outputs,
            expert_outputs=records,
            output_metadata=output_metadata,
            execution_metadata=execution_metadata,
            statistics=statistics,
            timing=TimingMetadata(),
            workspace=WorkspaceMetadata(True, active_workspace.estimated_bytes()),
            backend=self.config.backend,
            deterministic=self.config.deterministic,
        )


register_executor("triton", TritonExpertExecutor)
