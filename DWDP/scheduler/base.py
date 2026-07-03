from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from DWDP.dispatcher.plan import DispatchPlan

from .config import SchedulerConfig
from .execution import ExecutionPlan
from .workspace import SchedulerWorkspace


class BaseScheduler(nn.Module, ABC):
    """Abstract interface for execution schedulers."""

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        dispatch_plan: DispatchPlan,
        workspace: SchedulerWorkspace | None = None,
    ) -> ExecutionPlan:
        """Build an execution plan from a dispatch plan."""
