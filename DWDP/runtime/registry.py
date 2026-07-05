from __future__ import annotations

from .config import RuntimeConfig

_RUNTIME_REGISTRY: dict[str, type] = {}


def register_runtime(name: str, runtime_cls: type) -> None:
    """Register a runtime backend."""

    if name in _RUNTIME_REGISTRY:
        raise ValueError(f"Runtime backend '{name}' is already registered")
    _RUNTIME_REGISTRY[name] = runtime_cls


def get_runtime_class(name: str) -> type:
    """Resolve a runtime backend class."""

    try:
        return _RUNTIME_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown runtime backend '{name}'") from exc


def build_runtime(config: RuntimeConfig, *args, **kwargs):
    """Instantiate a registered runtime backend."""

    return get_runtime_class(config.backend)(config=config, *args, **kwargs)
