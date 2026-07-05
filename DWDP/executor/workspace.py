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
