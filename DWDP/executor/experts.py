from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ExpertBatch:
    """Contiguous expert-major batch passed to one expert module."""

    expert_id: int
    start: int
    end: int
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(slots=True)
class ExpertExecutionContext:
    """Execution context passed through the Executor backend."""

    expert_id: int
    priority: int
    stream_id: int
    deterministic: bool


class ExpertRegistry(nn.Module):
    """Container for expert modules keyed by global expert id."""

    def __init__(self, experts: Mapping[int, nn.Module] | Sequence[nn.Module]) -> None:
        super().__init__()
        self._key_to_name: dict[int, str] = {}

        if isinstance(experts, Mapping):
            items = sorted(experts.items(), key=lambda item: item[0])
        else:
            items = list(enumerate(experts))

        if not items:
            raise ValueError("ExpertRegistry requires at least one expert")

        for expert_id, module in items:
            if expert_id < 0:
                raise ValueError("expert ids must be non-negative")
            name = f"expert_{expert_id}"
            self.add_module(name, module)
            self._key_to_name[int(expert_id)] = name

    def get(self, expert_id: int) -> nn.Module:
        """Return the module for `expert_id`."""

        try:
            return getattr(self, self._key_to_name[int(expert_id)])
        except KeyError as exc:
            raise KeyError(f"Missing expert module for expert id {expert_id}") from exc

    @property
    def expert_ids(self) -> tuple[int, ...]:
        """Registered expert ids."""

        return tuple(sorted(self._key_to_name))
