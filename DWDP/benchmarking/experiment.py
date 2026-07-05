from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Filesystem layout for one benchmark experiment."""

    root: Path
    report_md: Path
    report_json: Path
    benchmark_config_json: Path
    environment_json: Path
    profiler_json: Path
    correctness_json: Path
    runtime_statistics_json: Path
    metadata_json: Path
    logs_dir: Path
    plots_dir: Path


def slugify(value: str) -> str:
    """Return a deterministic filesystem-safe slug."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    return slug.strip("_") or "experiment"


def create_experiment(
    *,
    results_root: str | Path = "results",
    model_name: str,
    backend: str,
    hardware: str | None = None,
    timestamp: datetime | None = None,
) -> ExperimentPaths:
    """Create a new timestamped benchmark experiment directory.

    The function never overwrites an existing experiment. If the timestamped
    name already exists, it appends a deterministic numeric suffix.
    """

    base = Path(results_root)
    active_timestamp = timestamp or datetime.now()
    stamp = active_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_".join(
        item
        for item in (
            slugify(model_name),
            slugify(backend),
            slugify(hardware) if hardware else None,
        )
        if item
    )
    candidate = base / f"{stamp}_{suffix}"
    index = 1
    root = candidate
    while root.exists():
        root = Path(f"{candidate}_{index:03d}")
        index += 1
    logs_dir = root / "logs"
    plots_dir = root / "plots"
    logs_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=False)
    return ExperimentPaths(
        root=root,
        report_md=root / "report.md",
        report_json=root / "report.json",
        benchmark_config_json=root / "benchmark_config.json",
        environment_json=root / "environment.json",
        profiler_json=root / "profiler.json",
        correctness_json=root / "correctness.json",
        runtime_statistics_json=root / "runtime_statistics.json",
        metadata_json=root / "metadata.json",
        logs_dir=logs_dir,
        plots_dir=plots_dir,
    )
