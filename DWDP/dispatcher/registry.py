from __future__ import annotations

from .config import DispatcherConfig

_DISPATCHER_REGISTRY: dict[str, type] = {}


def register_dispatcher(name: str, dispatcher_cls: type) -> None:
    """Register a dispatcher implementation."""

    if name in _DISPATCHER_REGISTRY:
        raise ValueError(f"Dispatcher '{name}' is already registered")
    _DISPATCHER_REGISTRY[name] = dispatcher_cls


def get_dispatcher_class(name: str) -> type:
    """Resolve a dispatcher implementation from the registry."""

    try:
        return _DISPATCHER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dispatcher type '{name}'") from exc


def build_dispatcher(config: DispatcherConfig) -> object:
    """Instantiate a dispatcher from configuration."""

    dispatcher_cls = get_dispatcher_class(config.dispatcher_type)
    return dispatcher_cls(config)
