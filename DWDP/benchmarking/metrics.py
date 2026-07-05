from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    """Memory metrics for one backend execution."""

    peak_gpu_memory_bytes: int | None = None
    average_gpu_memory_bytes: int | None = None
    peak_cpu_memory_bytes: int | None = None
    average_cpu_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class BackendPerformance:
    """End-to-end performance metrics for one backend."""

    backend: str
    ttft_ms: float | None = None
    prefill_latency_ms: float | None = None
    decode_latency_ms: float | None = None
    tokens_per_second: float | None = None
    total_runtime_ms: float | None = None
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)


@dataclass(frozen=True, slots=True)
class RuntimeBreakdown:
    """DWDP module timing breakdown."""

    router_ms: float | None = None
    dispatcher_ms: float | None = None
    scheduler_ms: float | None = None
    comms_planner_ms: float | None = None
    executor_ms: float | None = None
    merger_ms: float | None = None
    total_dwdp_overhead_ms: float | None = None
    module_percentages: dict[str, float] = field(default_factory=dict)
    pipeline_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerformanceComparison:
    """Fair benchmark comparison between Hugging Face and DWDP."""

    huggingface: BackendPerformance
    dwdp: BackendPerformance
    runtime_breakdown: RuntimeBreakdown = field(default_factory=RuntimeBreakdown)


@dataclass(frozen=True, slots=True)
class CorrectnessMetrics:
    """Correctness metrics comparing Hugging Face and DWDP."""

    max_absolute_error: float | None = None
    mean_absolute_error: float | None = None
    relative_error: float | None = None
    cosine_similarity: float | None = None
    torch_allclose: bool | None = None
    generated_token_parity: bool | None = None
    layer_output_parity: bool | None = None
    router_output_parity: bool | None = None
    executor_output_parity: bool | None = None
    merger_output_parity: bool | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatistics:
    """DWDP runtime statistics exported independently from performance metrics."""

    workspace_allocations: dict[str, int] = field(default_factory=dict)
    workspace_reuse: dict[str, bool] = field(default_factory=dict)
    num_experts: int | None = None
    active_experts: int | None = None
    routing_distribution: dict[str, int] = field(default_factory=dict)
    dispatcher_statistics: dict[str, object] = field(default_factory=dict)
    scheduler_statistics: dict[str, object] = field(default_factory=dict)
    communication_planner_statistics: dict[str, object] = field(default_factory=dict)
    executor_statistics: dict[str, object] = field(default_factory=dict)
    merger_statistics: dict[str, object] = field(default_factory=dict)
    future_extensions: dict[str, object] = field(default_factory=dict)
