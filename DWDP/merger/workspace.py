from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MergerWorkspace:
    """Reusable buffers for output reconstruction."""

    token_major_assignments: torch.Tensor | None = None
    merged_flat: torch.Tensor | None = None

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

    def get_assignment_buffer(
        self,
        num_assignments: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return reusable token-major assignment buffer."""

        return self._ensure_2d(
            "token_major_assignments",
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )

    def get_merged_buffer(
        self,
        num_tokens: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return reusable flat merged output buffer."""

        return self._ensure_2d(
            "merged_flat",
            num_tokens,
            output_size,
            dtype=dtype,
            device=device,
        )

    def estimated_bytes(self) -> int:
        """Estimate allocated workspace bytes."""

        total = 0
        for tensor in (self.token_major_assignments, self.merged_flat):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
