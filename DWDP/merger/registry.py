from __future__ import annotations

from .config import MergerConfig

_MERGER_REGISTRY: dict[str, type] = {}


def register_merger(name: str, merger_cls: type) -> None:
    """Register a merger backend."""

    if name in _MERGER_REGISTRY:
        raise ValueError(f"Merger backend '{name}' is already registered")
    _MERGER_REGISTRY[name] = merger_cls


def get_merger_class(name: str) -> type:
    """Resolve a merger backend class."""

    try:
        return _MERGER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown merger backend '{name}'") from exc


def build_merger(config: MergerConfig) -> object:
    """Instantiate a merger backend."""

    merger_cls = get_merger_class(config.backend)
    return merger_cls(config)
