"""Benchmark reporting infrastructure for reproducible DWDP experiments."""

from .config import BenchmarkConfig, GenerationConfig
from .environment import EnvironmentMetadata, collect_environment_metadata
from .experiment import ExperimentPaths, create_experiment
from .metrics import (
    BackendPerformance,
    CorrectnessMetrics,
    MemoryMetrics,
    PerformanceComparison,
    RuntimeBreakdown,
    RuntimeStatistics,
)
from .report import BenchmarkReport, ReportMetadata
from .writer import BenchmarkReportWriter

__all__ = [
    "BackendPerformance",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkReportWriter",
    "CorrectnessMetrics",
    "EnvironmentMetadata",
    "ExperimentPaths",
    "GenerationConfig",
    "MemoryMetrics",
    "PerformanceComparison",
    "ReportMetadata",
    "RuntimeBreakdown",
    "RuntimeStatistics",
    "collect_environment_metadata",
    "create_experiment",
]
