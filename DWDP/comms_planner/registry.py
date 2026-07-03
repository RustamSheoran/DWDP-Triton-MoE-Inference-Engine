from __future__ import annotations

from .config import CommunicationPlannerConfig

_COMMUNICATION_PLANNER_REGISTRY: dict[str, type] = {}


def register_communication_planner(name: str, planner_cls: type) -> None:
    """Register a communication planner implementation."""

    if name in _COMMUNICATION_PLANNER_REGISTRY:
        raise ValueError(f"Communication planner '{name}' is already registered")
    _COMMUNICATION_PLANNER_REGISTRY[name] = planner_cls


def get_communication_planner_class(name: str) -> type:
    """Resolve a communication planner class."""

    try:
        return _COMMUNICATION_PLANNER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown communication planner policy '{name}'") from exc


def build_communication_planner(config: CommunicationPlannerConfig) -> object:
    """Instantiate a communication planner from configuration."""

    planner_cls = get_communication_planner_class(config.planner_policy)
    return planner_cls(config)
