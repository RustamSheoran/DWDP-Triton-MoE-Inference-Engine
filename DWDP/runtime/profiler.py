from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class ModuleProfile:
    """Timing record for one DWDP runtime stage."""

    name: str
    duration_us: float


@dataclass(slots=True)
class RuntimeProfile:
    """Aggregated runtime profiling metadata."""

    module_profiles: tuple[ModuleProfile, ...]
    total_duration_us: float
    workspace_bytes: int = 0
    cuda_timing_available: bool = False
    nsight_placeholder: bool = True

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return a flat dictionary suitable for logging."""

        values: dict[str, float | int | bool] = {
            "total_duration_us": self.total_duration_us,
            "workspace_bytes": self.workspace_bytes,
            "cuda_timing_available": self.cuda_timing_available,
            "nsight_placeholder": self.nsight_placeholder,
        }
        for item in self.module_profiles:
            values[f"{item.name}_duration_us"] = item.duration_us
        return values


@dataclass(slots=True)
class RuntimeProfiler:
    """Low-overhead wall-clock profiler for reference runtime execution."""

    enabled: bool = False
    _records: list[ModuleProfile] = field(default_factory=list)
    _start_time: float | None = None

    def start(self) -> None:
        """Start total runtime timing."""

        self._records.clear()
        self._start_time = time.perf_counter() if self.enabled else None

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        """Record a stage duration when profiling is enabled."""

        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._records.append(ModuleProfile(name, (time.perf_counter() - start) * 1e6))

    def finish(self, workspace_bytes: int = 0) -> RuntimeProfile | None:
        """Finish profiling and return a runtime profile."""

        if not self.enabled or self._start_time is None:
            return None
        total = (time.perf_counter() - self._start_time) * 1e6
        return RuntimeProfile(
            module_profiles=tuple(self._records),
            total_duration_us=total,
            workspace_bytes=workspace_bytes,
        )
