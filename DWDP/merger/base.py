from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from DWDP.executor.outputs import ExecutorOutput

from .config import MergerConfig
from .outputs import MergerOutput
from .workspace import MergerWorkspace


class BaseMerger(nn.Module, ABC):
    """Abstract interface for output reconstruction backends."""

    def __init__(self, config: MergerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        executor_output: ExecutorOutput,
        workspace: MergerWorkspace | None = None,
    ) -> MergerOutput:
        """Reconstruct final hidden states from ExecutorOutput."""
