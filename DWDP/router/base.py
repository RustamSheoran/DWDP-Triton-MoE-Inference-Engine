from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .config import MetadataLevel, RouterConfig
from .types import RouterOutput


class BaseRouter(nn.Module, ABC):
    """Abstract base class for MoE routers."""

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def top_k(self) -> int:
        return self.config.top_k

    @abstractmethod
    def compute_router_logits(self, flat_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project flattened hidden states into expert logits."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata_level: MetadataLevel | None = None,
    ) -> RouterOutput:
        """Route tokens to experts."""
