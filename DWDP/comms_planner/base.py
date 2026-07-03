from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from DWDP.scheduler.execution import ExecutionPlan

from .config import CommunicationPlannerConfig
from .metadata import CommunicationPlan
from .workspace import CommunicationPlannerWorkspace


class BaseCommunicationPlanner(nn.Module, ABC):
    """Abstract interface for communication planners."""

    def __init__(self, config: CommunicationPlannerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        execution_plan: ExecutionPlan,
        workspace: CommunicationPlannerWorkspace | None = None,
    ) -> CommunicationPlan:
        """Build a communication plan from an execution plan."""
