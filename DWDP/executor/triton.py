"""Triton executor backend boundary.

No Triton kernels are launched yet. This class validates and owns the
storage-preserving weight provider needed by future grouped GEMM kernels, then
uses the PyTorch backend as a correctness fallback.
"""

from __future__ import annotations

import torch

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .config import ExecutorConfig
from .experts import ExpertRegistry
from .extractors import extract_qwen_swiglu_weight_provider
from .outputs import ExecutorOutput
from .pytorch import PyTorchExecutor
from .registry import register_executor
from .weights import ExpertWeightProvider
from .workspace import ExecutorWorkspace


class TritonExpertExecutor(PyTorchExecutor):
    """Grouped-expert backend skeleton with a PyTorch correctness fallback.

    The provider preserves original Qwen parameter storage and defines the
    future grouped-GEMM ABI. Until Triton kernels are added, ``forward``
    intentionally delegates to ``PyTorchExecutor``.
    """

    def __init__(
        self,
        config: ExecutorConfig,
        experts: ExpertRegistry,
        weight_provider: ExpertWeightProvider | None = None,
    ) -> None:
        super().__init__(config, experts)
        self.weight_provider = weight_provider or extract_qwen_swiglu_weight_provider(experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None = None,
    ) -> ExecutorOutput:
        """Run the unchanged reference path while preserving the future ABI."""

        output = super().forward(
            hidden_states,
            dispatch_plan,
            execution_plan,
            communication_plan,
            workspace=workspace,
        )
        output.backend = "triton_reference_fallback"
        output.statistics.backend = "triton_reference_fallback"
        return output


register_executor("triton", TritonExpertExecutor)
