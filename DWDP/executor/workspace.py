from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ExecutorWorkspace:
    """Reusable buffers for reference expert execution."""

    packed_expert_outputs: torch.Tensor | None = None
    weighted_expert_outputs: torch.Tensor | None = None
    gathered_activations: torch.Tensor | None = None
    temporary_outputs: torch.Tensor | None = None
    # The scheduler stores its compact execution description on the device.
    # Keep the last host copy here so repeated decode shapes do not force a
    # device-to-host read before every expert launch.
    _schedule_tensors: tuple[torch.Tensor, ...] | None = None
    _schedule_rows: tuple[tuple[int, int, int, int, int, int], ...] | None = None

    def _ensure_2d(
        self,
        name: str,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = getattr(self, name)
        if (
            tensor is None
            or tensor.shape[0] < rows
            or tensor.shape[1] != cols
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            tensor = torch.empty(rows, cols, dtype=dtype, device=device)
            setattr(self, name, tensor)
        return tensor[:rows, :cols]

    def get_output_buffers(
        self,
        num_assignments: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed and weighted output buffers."""

        packed = self._ensure_2d(
            "packed_expert_outputs",
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )
        weighted = self._ensure_2d(
            "weighted_expert_outputs",
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )
        return packed, weighted

    def get_gather_buffer(
        self,
        rows: int,
        hidden_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a reusable gathered activation buffer."""

        return self._ensure_2d(
            "gathered_activations",
            rows,
            hidden_size,
            dtype=dtype,
            device=device,
        )

    def get_schedule_rows(
        self,
        expert_queue: torch.Tensor,
        expert_starts: torch.Tensor,
        expert_ends: torch.Tensor,
        expert_counts: torch.Tensor,
        execution_priority: torch.Tensor,
        stream_assignments: torch.Tensor,
    ) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return host execution rows with one device synchronization at most.

        Calling ``Tensor.item()`` for every scheduler field forces CUDA to
        synchronize per scalar.  Materializing the small schedule once makes
        the host loop independent of device scalar reads.  Holding tensor
        references, rather than a pointer-only key, keeps the cache safe when
        the allocator recycles storage for a later plan.
        """

        tensors = (
            expert_queue,
            expert_starts,
            expert_ends,
            expert_counts,
            execution_priority,
            stream_assignments,
        )
        if (
            self._schedule_tensors is not None
            and all(cached is current for cached, current in zip(self._schedule_tensors, tensors))
            and self._schedule_rows is not None
        ):
            return self._schedule_rows

        # stack performs one compact device copy before tolist transfers the
        # complete schedule, replacing six scalar D2H synchronizations per
        # active expert.
        values = torch.stack(tensors).cpu().tolist()
        rows = tuple(zip(*values))
        self._schedule_tensors = tensors
        self._schedule_rows = rows
        return rows

    def estimated_bytes(self) -> int:
        """Estimate allocated workspace bytes."""

        total = 0
        for tensor in (
            self.packed_expert_outputs,
            self.weighted_expert_outputs,
            self.gathered_activations,
            self.temporary_outputs,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
