from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from DWDP.router.types import RouterOutput

from .config import DispatcherConfig
from .plan import DispatchPlan
from .workspace import DispatchWorkspace


class BaseDispatcher(nn.Module, ABC):
    """Abstract interface for dispatch planners."""

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @abstractmethod
    def forward(
        self,
        router_output: RouterOutput,
        workspace: DispatchWorkspace | None = None,
    ) -> DispatchPlan:
        """Build a dispatch plan from completed routing output."""
