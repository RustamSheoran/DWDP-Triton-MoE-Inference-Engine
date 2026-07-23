from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import BenchmarkConfig
from .environment import EnvironmentMetadata
from .metrics import CorrectnessMetrics, PerformanceComparison, RuntimeStatistics


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Metadata identifying one benchmark experiment."""

    experiment_name: str
    schema_version: str = "1.0"
    notes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    future_extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete benchmark report payload."""

    metadata: ReportMetadata
    config: BenchmarkConfig
    environment: EnvironmentMetadata
    performance: PerformanceComparison
    correctness: CorrectnessMetrics
    runtime_statistics: RuntimeStatistics
    profiler: dict[str, Any] = field(default_factory=dict)
    observations: tuple[str, ...] = ()


def _fmt(value: object, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def _bool(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "yes" if value else "no"


def _change_pct(candidate: float | int | None, reference: float | int | None) -> str:
    """Format candidate-vs-reference percentage change."""

    if candidate is None or reference in (None, 0):
        return "N/A"
    return f"{(float(candidate) / float(reference) - 1.0) * 100.0:+.2f}%"


def _profile_rows(payload: object) -> list[tuple[str, float | None, float | None, str]]:
    if not isinstance(payload, dict):
        return []
    rows: list[tuple[str, float | None, float | None, str]] = []
    for name, value in payload.items():
        if name == "top_operators" or not isinstance(value, dict):
            continue
        operators = value.get("operators", ())
        rows.append((name, value.get("cpu_ms"), value.get("device_ms"), ", ".join(operators)))
    return rows


def render_markdown(report: BenchmarkReport) -> str:
    """Render a human-readable Markdown benchmark report."""

    cfg = report.config
    env = report.environment
    perf = report.performance
    correctness = report.correctness
    breakdown = perf.runtime_breakdown
    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"- Experiment: `{report.metadata.experiment_name}`")
    lines.append(f"- Model: `{cfg.model_name}`")
    lines.append(f"- Checkpoint: `{cfg.checkpoint}`")
    lines.append(f"- Backend comparison: `{perf.huggingface.backend}` vs `{perf.dwdp.backend}`")
    lines.append(f"- Timestamp: `{env.benchmark_timestamp}`")
    lines.append("")
    lines.append("# Environment")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key, value in (
        ("GPU", env.gpu_model),
        ("GPU Memory", env.gpu_memory_bytes),
        ("CUDA", env.cuda_version),
        ("cuDNN", env.cudnn_version),
        ("PyTorch", env.pytorch_version),
        ("Transformers", env.transformers_version),
        ("Triton", env.triton_version),
        ("NVIDIA Driver", env.nvidia_driver_version),
        ("Python", env.python_version),
        ("OS", env.operating_system),
        ("Git Commit", env.git_commit_hash),
        ("Git Branch", env.git_branch),
        ("Runtime Backend", env.runtime_backend),
        ("Precision", env.precision),
        ("Torch Compile", env.torch_compile),
    ):
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    lines.append("# Configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Prompt | `{cfg.prompt}` |")
    lines.append(f"| Batch Size | {cfg.batch_size} |")
    lines.append(f"| Sequence Length | {cfg.sequence_length} |")
    lines.append(f"| Max New Tokens | {cfg.generation.max_new_tokens} |")
    lines.append(f"| Temperature | {cfg.generation.temperature} |")
    lines.append(f"| Top-k | {_fmt(cfg.generation.top_k)} |")
    lines.append(f"| Top-p | {_fmt(cfg.generation.top_p)} |")
    lines.append(f"| DType | {cfg.dtype} |")
    lines.append(f"| Device | {cfg.device} |")
    lines.append(f"| Random Seed | {cfg.random_seed} |")
    lines.append(f"| Workspace | {_bool(cfg.workspace_enabled)} |")
    lines.append("")
    lines.append("# Performance Results")
    lines.append("")
    lines.append("| Backend | TTFT ms | Prefill ms | Decode ms | Tokens/s | Total ms | Peak GPU bytes |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in (perf.huggingface, perf.dwdp):
        lines.append(
            f"| {item.backend} | {_fmt(item.ttft_ms)} | {_fmt(item.prefill_latency_ms)} | "
            f"{_fmt(item.decode_latency_ms)} | {_fmt(item.tokens_per_second)} | {_fmt(item.total_runtime_ms)} | "
            f"{_fmt(item.memory.peak_gpu_memory_bytes)} |"
        )
    lines.append("")
    lines.append("## DWDP vs Native Hugging Face")
    lines.append("")
    lines.append("| Metric | Native HF | DWDP | DWDP change |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label, reference, candidate in (
        ("TTFT ms", perf.huggingface.ttft_ms, perf.dwdp.ttft_ms),
        ("Prefill ms", perf.huggingface.prefill_latency_ms, perf.dwdp.prefill_latency_ms),
        ("Decode ms", perf.huggingface.decode_latency_ms, perf.dwdp.decode_latency_ms),
        ("Tokens/s", perf.huggingface.tokens_per_second, perf.dwdp.tokens_per_second),
        ("Total latency ms", perf.huggingface.total_runtime_ms, perf.dwdp.total_runtime_ms),
        (
            "Peak GPU memory bytes",
            perf.huggingface.memory.peak_gpu_memory_bytes,
            perf.dwdp.memory.peak_gpu_memory_bytes,
        ),
    ):
        lines.append(f"| {label} | {_fmt(reference)} | {_fmt(candidate)} | {_change_pct(candidate, reference)} |")
    latency_change = _change_pct(perf.dwdp.total_runtime_ms, perf.huggingface.total_runtime_ms)
    throughput_change = _change_pct(perf.dwdp.tokens_per_second, perf.huggingface.tokens_per_second)
    if latency_change != "N/A":
        direction = "faster" if float(latency_change.strip("+%")) < 0 else "slower"
        lines.append("")
        lines.append(f"**Summary:** DWDP is {abs(float(latency_change.strip('+%'))):.2f}% {direction} than native HF by end-to-end latency.")
        lines.append(f"DWDP throughput is {throughput_change} versus native HF.")
    lines.append("")
    lines.append("# Runtime Breakdown")
    lines.append("")
    lines.append("| Module | Latency ms | Percentage |")
    lines.append("| --- | ---: | ---: |")
    for name, value in (
        ("Router", breakdown.router_ms),
        ("Dispatcher", breakdown.dispatcher_ms),
        ("Scheduler", breakdown.scheduler_ms),
        ("Comms Planner", breakdown.comms_planner_ms),
        ("Executor", breakdown.executor_ms),
        ("Merger", breakdown.merger_ms),
    ):
        pct = breakdown.module_percentages.get(name.lower().replace(" ", "_"))
        lines.append(f"| {name} | {_fmt(value)} | {_fmt(pct, '%')} |")
    lines.append(f"| Total DWDP Overhead | {_fmt(breakdown.total_dwdp_overhead_ms)} | N/A |")
    lines.append("")
    lines.append("# Correctness Validation")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Maximum Absolute Error | {_fmt(correctness.max_absolute_error)} |")
    lines.append(f"| Mean Absolute Error | {_fmt(correctness.mean_absolute_error)} |")
    lines.append(f"| Relative Error | {_fmt(correctness.relative_error)} |")
    lines.append(f"| Cosine Similarity | {_fmt(correctness.cosine_similarity)} |")
    lines.append(f"| torch.allclose | {_bool(correctness.torch_allclose)} |")
    lines.append(f"| Generated Token Parity | {_bool(correctness.generated_token_parity)} |")
    lines.append(f"| Layer Output Parity | {_bool(correctness.layer_output_parity)} |")
    lines.append(f"| Router Output Parity | {_bool(correctness.router_output_parity)} |")
    lines.append(f"| Executor Output Parity | {_bool(correctness.executor_output_parity)} |")
    lines.append(f"| Merger Output Parity | {_bool(correctness.merger_output_parity)} |")
    lines.append("")
    lines.append("# Memory Usage")
    lines.append("")
    lines.append("| Backend | Peak GPU Bytes | Average GPU Bytes |")
    lines.append("| --- | ---: | ---: |")
    for item in (perf.huggingface, perf.dwdp):
        lines.append(
            f"| {item.backend} | {_fmt(item.memory.peak_gpu_memory_bytes)} | "
            f"{_fmt(item.memory.average_gpu_memory_bytes)} |"
        )
    lines.append("")
    lines.append("# Profiling Summary")
    lines.append("")
    if report.profiler:
        lines.append("Load and profiler configuration:")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | ---: |")
        for key in ("hf_load_time_ms", "dwdp_load_time_ms", "torch_profiler_enabled"):
            if key in report.profiler:
                lines.append(f"| {key} | {_fmt(report.profiler[key])} |")
        for backend in ("hf", "dwdp"):
            rows = _profile_rows(report.profiler.get(backend))
            if not rows:
                continue
            lines.append("")
            lines.append(f"### {backend.upper()} operator categories")
            lines.append("")
            lines.append("| Category | CPU self ms | Device self ms | Operators |")
            lines.append("| --- | ---: | ---: | --- |")
            for name, cpu_ms, device_ms, operators in rows:
                lines.append(f"| {name} | {_fmt(cpu_ms)} | {_fmt(device_ms)} | {operators or 'N/A'} |")
            top_operators = report.profiler.get(backend, {}).get("top_operators", [])
            if top_operators:
                lines.append("")
                lines.append("Top operators by CPU self time:")
                lines.append("")
                lines.append("| Operator | CPU self ms | Device self ms | Calls |")
                lines.append("| --- | ---: | ---: | ---: |")
                for item in top_operators[:10]:
                    lines.append(
                        f"| {item.get('operator', 'N/A')} | {_fmt(item.get('self_cpu_ms'))} | "
                        f"{_fmt(item.get('self_device_ms'))} | {_fmt(item.get('calls'))} |"
                    )
    else:
        lines.append("No profiler payload was provided.")
    lines.append("")
    lines.append("# Notes")
    lines.append("")
    notes = (*report.metadata.notes, *report.observations)
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No notes recorded.")
    lines.append("")
    return "\n".join(lines)
