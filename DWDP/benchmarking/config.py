from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation settings shared by Hugging Face and DWDP benchmarks."""

    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be >= 0 when provided")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Complete reproducibility configuration for one benchmark experiment."""

    model_name: str
    checkpoint: str
    prompt: str
    batch_size: int
    sequence_length: int
    generation: GenerationConfig
    dtype: str
    device: str
    random_seed: int
    backend: str = "hf"
    compare_backend: str = "dwdp"
    torch_compile: bool = False
    workspace_enabled: bool = True
    compile_mode: str | None = None
    runtime_backend: str = "dwdp"
    runtime_config: dict[str, Any] = field(default_factory=dict)
    sampling_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not self.device:
            raise ValueError("device must be non-empty")
