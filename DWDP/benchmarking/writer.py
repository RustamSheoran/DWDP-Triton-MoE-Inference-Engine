from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment import ExperimentPaths, create_experiment
from .report import BenchmarkReport, render_markdown
from .serialization import to_jsonable, write_json


@dataclass(slots=True)
class BenchmarkReportWriter:
    """Writes benchmark reports and sidecar artifacts to disk."""

    results_root: str | Path = "results"

    def create_paths(self, report: BenchmarkReport) -> ExperimentPaths:
        """Create a fresh experiment directory for `report`."""

        hardware = report.environment.gpu_model or "unknown_hw"
        return create_experiment(
            results_root=self.results_root,
            model_name=report.config.model_name,
            backend=f"{report.performance.huggingface.backend}_vs_{report.performance.dwdp.backend}",
            hardware=hardware,
        )

    def write(self, report: BenchmarkReport) -> ExperimentPaths:
        """Write all required report artifacts and return their paths."""

        paths = self.create_paths(report)
        paths.report_md.write_text(render_markdown(report), encoding="utf-8")
        write_json(paths.report_json, report)
        write_json(paths.benchmark_config_json, report.config)
        write_json(paths.environment_json, report.environment)
        write_json(paths.profiler_json, report.profiler)
        write_json(paths.correctness_json, report.correctness)
        write_json(paths.runtime_statistics_json, report.runtime_statistics)
        write_json(
            paths.metadata_json,
            {
                "metadata": to_jsonable(report.metadata),
                "schema_version": report.metadata.schema_version,
                "artifact_files": {
                    "report_md": paths.report_md.name,
                    "report_json": paths.report_json.name,
                    "benchmark_config_json": paths.benchmark_config_json.name,
                    "environment_json": paths.environment_json.name,
                    "profiler_json": paths.profiler_json.name,
                    "correctness_json": paths.correctness_json.name,
                    "runtime_statistics_json": paths.runtime_statistics_json.name,
                },
                "future_extensions": {
                    "plots_dir": paths.plots_dir.name,
                    "logs_dir": paths.logs_dir.name,
                    "nsight_exports": None,
                    "multi_gpu_topology": None,
                    "power_metrics": None,
                    "energy_efficiency": None,
                },
            },
        )
        return paths
