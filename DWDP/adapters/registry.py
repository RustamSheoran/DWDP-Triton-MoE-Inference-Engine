from __future__ import annotations

from typing import Any

from DWDP.runtime.config import RuntimeConfig

_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str, adapter_cls: type) -> None:
    """Register a model adapter."""

    if name in _ADAPTER_REGISTRY:
        raise ValueError(f"Adapter '{name}' is already registered")
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter_class(name: str) -> type:
    """Resolve an adapter class."""

    try:
        return _ADAPTER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter '{name}'") from exc


def build_adapter(name: str, *, model: Any | None = None, tokenizer: Any | None = None, config: RuntimeConfig | None = None):
    """Instantiate a registered adapter."""

    return get_adapter_class(name)(model=model, tokenizer=tokenizer, config=config)
