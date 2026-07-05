from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from DWDP.executor import ExpertRegistry
from DWDP.runtime.config import RuntimeConfig

from .base import BaseModelAdapter
from .registry import register_adapter


class HuggingFaceAdapter(BaseModelAdapter):
    """Hugging Face integration boundary for DWDP.

    The reference adapter preserves the native Hugging Face model for all
    non-MoE behavior. Explicit MoE component binding is supported through
    `bind_moe_layer`; model-specific automatic patching is intentionally left
    to future Qwen/Mixtral/DeepSeek adapters.
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, config: RuntimeConfig | None = None, **kwargs) -> "HuggingFaceAdapter":
        """Load a Hugging Face causal LM and optional tokenizer."""

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HuggingFaceAdapter.from_pretrained requires transformers") from exc

        runtime_config = config or RuntimeConfig()
        model_kwargs = dict(kwargs)
        tokenizer = model_kwargs.pop("tokenizer", None)
        load_tokenizer = bool(model_kwargs.pop("load_tokenizer", True))
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if load_tokenizer and tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model=model, tokenizer=tokenizer, config=runtime_config)

    def create_runtime(self):
        """Create a runtime wrapper around the Hugging Face model.

        If an MoE layer has been explicitly bound, the runtime owns the DWDP
        modules for that layer. Otherwise the runtime acts as a HF-compatible
        wrapper and delegates generation to the native model.
        """

        from DWDP.runtime.runtime import DWDPRuntime

        binding = getattr(self, "_moe_binding", None)
        if binding is None:
            return _DelegatingRuntime(adapter=self, config=self.config)
        return DWDPRuntime.build_reference(
            hidden_size=binding["hidden_size"],
            num_experts=binding["num_experts"],
            top_k=binding["top_k"],
            experts=binding["experts"],
            router=binding.get("router"),
            config=self.config,
        )

    def bind_moe_layer(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        experts: Mapping[int, nn.Module] | Sequence[nn.Module] | ExpertRegistry,
        router: nn.Module | None = None,
    ) -> "HuggingFaceAdapter":
        """Bind one explicit MoE layer to DWDP reference execution."""

        self._moe_binding = {
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "experts": experts,
            "router": router,
        }
        return self


class _DelegatingRuntime(nn.Module):
    """HF-compatible wrapper used before model-specific MoE patching exists."""

    def __init__(self, *, adapter: HuggingFaceAdapter, config: RuntimeConfig) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config
        if adapter.model is not None:
            self.model = adapter.model

    def forward(self, *args, **kwargs):
        """Delegate model forward to Hugging Face."""

        return self.adapter.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generation to Hugging Face."""

        return self.adapter.generate(*args, **kwargs)

    def compile(self):
        """Return self; native model compilation is left to the caller."""

        return self

    def profile(self, *args, **kwargs):
        """Profile placeholders are unavailable for a pure delegating wrapper."""

        del args, kwargs
        return None

    def benchmark(self, *args, **kwargs):
        """Benchmarking for a delegating wrapper is handled by CLI harnesses."""

        raise RuntimeError("DWDP module benchmark requires an explicitly bound MoE layer")


register_adapter("huggingface", HuggingFaceAdapter)
