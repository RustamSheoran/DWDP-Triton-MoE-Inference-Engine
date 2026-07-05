from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from DWDP.comms_planner.metadata import CommunicationPlan
from DWDP.dispatcher.plan import DispatchPlan
from DWDP.scheduler.execution import ExecutionPlan

from .config import ExecutorConfig
from .experts import ExpertRegistry
from .outputs import ExecutorOutput
from .workspace import ExecutorWorkspace


class BaseExecutor(nn.Module, ABC):
    """Abstract interface for expert execution backends."""

    def __init__(self, config: ExecutorConfig, experts: ExpertRegistry) -> None:
        super().__init__()
        self.config = config
        self.experts = experts

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatch_plan: DispatchPlan,
        execution_plan: ExecutionPlan,
        communication_plan: CommunicationPlan,
        workspace: ExecutorWorkspace | None = None,
    ) -> ExecutorOutput:
        """Execute experts according to finalized runtime plans."""
