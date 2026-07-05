from __future__ import annotations

import json
from datetime import datetime

from DWDP.benchmarking import (
    BackendPerformance,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkReportWriter,
    CorrectnessMetrics,
    EnvironmentMetadata,
    GenerationConfig,
    MemoryMetrics,
    PerformanceComparison,
    ReportMetadata,
    RuntimeBreakdown,
    RuntimeStatistics,
    create_experiment,
)
from DWDP.benchmarking.report import render_markdown


def make_report() -> BenchmarkReport:
    return BenchmarkReport(
        metadata=ReportMetadata(
            experiment_name="qwen15moe_a2.7b_h100",
            notes=("reference run",),
        ),
        config=BenchmarkConfig(
            model_name="Qwen/Qwen1.5-MoE-A2.7B",
            checkpoint="Qwen/Qwen1.5-MoE-A2.7B",
            prompt="Hello",
            batch_size=1,
            sequence_length=16,
            generation=GenerationConfig(max_new_tokens=8, temperature=1.0),
            dtype="bf16",
            device="cuda",
            random_seed=1234,
        ),
        environment=EnvironmentMetadata(
            benchmark_timestamp="2026-07-05T09:02:18+00:00",
            python_version="3.12",
            operating_system="Linux",
            hostname="host",
            git_commit_hash="abc123",
            git_branch="main",
            pytorch_version="2.x",
            transformers_version="4.x",
            triton_version=None,
            cuda_version="12.x",
            cudnn_version="9.x",
            nvidia_driver_version="555",
            gpu_model="H100",
            gpu_memory_bytes=80_000_000_000,
            runtime_backend="dwdp",
            precision="bf16",
            torch_compile=False,
        ),
        performance=PerformanceComparison(
            huggingface=BackendPerformance(
                backend="hf",
                ttft_ms=10.0,
                prefill_latency_ms=9.0,
                decode_latency_ms=5.0,
                tokens_per_second=100.0,
                total_runtime_ms=20.0,
                memory=MemoryMetrics(peak_gpu_memory_bytes=100),
            ),
            dwdp=BackendPerformance(
                backend="dwdp",
                ttft_ms=11.0,
                prefill_latency_ms=10.0,
                decode_latency_ms=6.0,
                tokens_per_second=90.0,
                total_runtime_ms=22.0,
                memory=MemoryMetrics(peak_gpu_memory_bytes=120),
            ),
            runtime_breakdown=RuntimeBreakdown(
                router_ms=1.0,
                dispatcher_ms=2.0,
                scheduler_ms=0.5,
                comms_planner_ms=0.1,
                executor_ms=10.0,
                merger_ms=1.5,
                total_dwdp_overhead_ms=5.1,
                module_percentages={"router": 10.0},
            ),
        ),
        correctness=CorrectnessMetrics(
            max_absolute_error=1e-4,
            mean_absolute_error=1e-5,
            relative_error=1e-4,
            cosine_similarity=0.9999,
            torch_allclose=True,
            generated_token_parity=True,
        ),
        runtime_statistics=RuntimeStatistics(
            workspace_allocations={"total_bytes": 4096},
            num_experts=60,
            active_experts=8,
        ),
        profiler={"router_duration_us": 100.0},
    )


def test_create_experiment_never_overwrites(tmp_path) -> None:
    timestamp = datetime(2026, 7, 5, 14, 32, 18)

    first = create_experiment(
        results_root=tmp_path,
        model_name="qwen15moe_a2.7b",
        backend="dwdp",
        hardware="h100",
        timestamp=timestamp,
    )
    second = create_experiment(
        results_root=tmp_path,
        model_name="qwen15moe_a2.7b",
        backend="dwdp",
        hardware="h100",
        timestamp=timestamp,
    )

    assert first.root != second.root
    assert first.root.name == "2026-07-05_14-32-18_qwen15moe_a2.7b_dwdp_h100"
    assert second.root.name.endswith("_001")
    assert first.logs_dir.exists()
    assert first.plots_dir.exists()


def test_markdown_report_contains_required_sections() -> None:
    markdown = render_markdown(make_report())

    for section in (
        "# Benchmark Summary",
        "# Environment",
        "# Configuration",
        "# Performance Results",
        "# Runtime Breakdown",
        "# Correctness Validation",
        "# Memory Usage",
        "# Profiling Summary",
        "# Notes",
    ):
        assert section in markdown


def test_report_writer_emits_all_required_artifacts(tmp_path) -> None:
    writer = BenchmarkReportWriter(results_root=tmp_path)

    paths = writer.write(make_report())

    for path in (
        paths.report_md,
        paths.report_json,
        paths.benchmark_config_json,
        paths.environment_json,
        paths.profiler_json,
        paths.correctness_json,
        paths.runtime_statistics_json,
        paths.metadata_json,
    ):
        assert path.exists()
    assert paths.logs_dir.exists()
    assert paths.plots_dir.exists()
    report = json.loads(paths.report_json.read_text(encoding="utf-8"))
    assert report["config"]["model_name"] == "Qwen/Qwen1.5-MoE-A2.7B"
    assert report["correctness"]["generated_token_parity"]
