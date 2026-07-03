from __future__ import annotations

from .config import SchedulerConfig

_SCHEDULER_REGISTRY: dict[str, type] = {}


def register_scheduler(name: str, scheduler_cls: type) -> None:
    """Register a scheduler implementation."""

    if name in _SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' is already registered")
    _SCHEDULER_REGISTRY[name] = scheduler_cls


def get_scheduler_class(name: str) -> type:
    """Resolve a scheduler class from the registry."""

    try:
        return _SCHEDULER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scheduler policy '{name}'") from exc


def build_scheduler(config: SchedulerConfig) -> object:
    """Instantiate a scheduler from configuration."""

    scheduler_cls = get_scheduler_class(config.scheduling_policy)
    return scheduler_cls(config)
