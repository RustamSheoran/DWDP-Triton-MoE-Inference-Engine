from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from DWDP.runtime.config import RuntimeConfig


class BaseModelAdapter(ABC):
    """Adapter boundary between external model frameworks and DWDP runtime."""

    def __init__(self, *, model: Any | None = None, tokenizer: Any | None = None, config: RuntimeConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RuntimeConfig()

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: str, *, config: RuntimeConfig | None = None, **kwargs) -> "BaseModelAdapter":
        """Load an external model and return an initialized adapter."""

    @abstractmethod
    def create_runtime(self):
        """Create a `DWDPRuntime` instance for this adapter."""

    def forward(self, *args, **kwargs):
        """Forward to the wrapped model outside explicit MoE layer execution."""

        if self.model is None:
            raise RuntimeError("adapter has no wrapped model")
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generation to the wrapped model/tokenizer stack."""

        if self.model is None or not hasattr(self.model, "generate"):
            raise RuntimeError("adapter model does not expose generate()")
        return self.model.generate(*args, **kwargs)
