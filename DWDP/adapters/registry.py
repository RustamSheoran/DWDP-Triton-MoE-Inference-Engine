from __future__ import annotations

from typing import Any

from DWDP.runtime.config import RuntimeConfig

_ADAPTER_REGISTRY: dict[str, type] = {}
_MODEL_ADAPTER_REGISTRY: list[tuple[tuple[str, ...], type]] = []


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


def register_model_adapter(model_types: tuple[str, ...], adapter_cls: type) -> None:
    """Register an adapter for Hugging Face config/model type detection."""

    _MODEL_ADAPTER_REGISTRY.append((model_types, adapter_cls))


def detect_adapter_class(model: Any) -> type | None:
    """Return the first registered adapter matching a Hugging Face model."""

    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = tuple(str(item).lower() for item in getattr(config, "architectures", ()) or ())
    model_class = type(model).__name__.lower()
    candidates = (model_type, model_class, *architectures)
    for patterns, adapter_cls in _MODEL_ADAPTER_REGISTRY:
        if any(pattern.lower() in candidate for pattern in patterns for candidate in candidates):
            return adapter_cls
    return None
