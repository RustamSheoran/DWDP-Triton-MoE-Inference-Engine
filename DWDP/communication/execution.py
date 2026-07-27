"""PyTorch runtime bridge for the compiled DWDP communication engine ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from DWDP.executor.experts import ExpertRegistry


@dataclass(frozen=True, slots=True)
class ExpertPointer:
    """Executable expert reference resolved only through the communication layer."""

    module: nn.Module
    device_pointers: tuple[int, ...]


class ExecutionCommunicationEngine(nn.Module):
    """Communication-engine boundary used by the Python reference executor."""

    def __init__(self, experts: ExpertRegistry) -> None:
        super().__init__()
        self._registry = experts

    def getWeight(self, expert_id: int) -> ExpertPointer:
        module = self._registry.get(expert_id)
        pointers = tuple(
            parameter.data_ptr()
            for parameter in module.parameters(recurse=True)
            if parameter.device.type == "cuda"
        )
        return ExpertPointer(module=module, device_pointers=pointers)

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._registry.expert_ids
