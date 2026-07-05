from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from .config import ExecutorConfig
from .experts import ExpertRegistry

_EXECUTOR_REGISTRY: dict[str, type] = {}


def register_executor(name: str, executor_cls: type) -> None:
    """Register an executor backend."""

    if name in _EXECUTOR_REGISTRY:
        raise ValueError(f"Executor backend '{name}' is already registered")
    _EXECUTOR_REGISTRY[name] = executor_cls


def get_executor_class(name: str) -> type:
    """Resolve an executor backend class."""

    try:
        return _EXECUTOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown executor backend '{name}'") from exc


def build_executor(
    config: ExecutorConfig,
    experts: ExpertRegistry | Mapping[int, nn.Module] | Sequence[nn.Module],
) -> object:
    """Instantiate an executor from configuration and expert modules."""

    executor_cls = get_executor_class(config.backend)
    registry = experts if isinstance(experts, ExpertRegistry) else ExpertRegistry(experts)
    return executor_cls(config, registry)
