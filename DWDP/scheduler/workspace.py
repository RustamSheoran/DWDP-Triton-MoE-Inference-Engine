from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SchedulerWorkspace:
    """Reusable scheduler buffers for per-iteration planning."""

    execution_order: torch.Tensor | None = None
    expert_queue: torch.Tensor | None = None
    expert_starts: torch.Tensor | None = None
    expert_ends: torch.Tensor | None = None
    expert_counts: torch.Tensor | None = None
    execution_priority: torch.Tensor | None = None
    stream_assignments: torch.Tensor | None = None
    barrier_after_batch: torch.Tensor | None = None
    dependency_src: torch.Tensor | None = None
    dependency_dst: torch.Tensor | None = None

    def _ensure_1d(
        self,
        name: str,
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = getattr(self, name)
        if (
            tensor is None
            or tensor.numel() < size
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            tensor = torch.empty(size, dtype=dtype, device=device)
            setattr(self, name, tensor)
        return tensor[:size]

    def get_active_expert_buffers(
        self,
        active_count: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reusable buffers sized by active expert count."""

        return (
            self._ensure_1d("execution_order", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("expert_queue", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("expert_starts", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("expert_ends", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("expert_counts", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("execution_priority", active_count, dtype=torch.int64, device=device),
            self._ensure_1d("stream_assignments", active_count, dtype=torch.int64, device=device),
        )

    def get_barrier_buffer(
        self,
        active_count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return reusable boolean barrier buffer."""

        return self._ensure_1d(
            "barrier_after_batch",
            active_count,
            dtype=torch.bool,
            device=device,
        )

    def get_dependency_buffers(
        self,
        dependency_count: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable dependency edge buffers."""

        return (
            self._ensure_1d("dependency_src", dependency_count, dtype=torch.int64, device=device),
            self._ensure_1d("dependency_dst", dependency_count, dtype=torch.int64, device=device),
        )

    def estimated_bytes(self) -> int:
        """Estimate total allocated workspace bytes."""

        total = 0
        for tensor in (
            self.execution_order,
            self.expert_queue,
            self.expert_starts,
            self.expert_ends,
            self.expert_counts,
            self.execution_priority,
            self.stream_assignments,
            self.barrier_after_batch,
            self.dependency_src,
            self.dependency_dst,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
