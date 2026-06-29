from __future__ import annotations

from collections.abc import Callable

from .config import RouterConfig

RouterFactory = Callable[[RouterConfig], object]

_ROUTER_REGISTRY: dict[str, type] = {}


def register_router(name: str, router_cls: type) -> None:
    """Register a router class under a stable name."""

    if name in _ROUTER_REGISTRY:
        raise ValueError(f"Router '{name}' is already registered")
    _ROUTER_REGISTRY[name] = router_cls


def get_router_class(name: str) -> type:
    """Look up a registered router class."""

    try:
        return _ROUTER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown router type '{name}'") from exc


def build_router(config: RouterConfig) -> object:
    """Instantiate a router from its config."""

    router_cls = get_router_class(config.router_type)
    return router_cls(config)
