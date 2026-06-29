from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DispatchWorkspace:
    """Reusable workspace buffers for dispatch planning."""

    token_permutation: torch.Tensor | None = None
    inverse_permutation: torch.Tensor | None = None
    packed_expert_ids: torch.Tensor | None = None
    packed_token_indices: torch.Tensor | None = None
    packed_routing_weights: torch.Tensor | None = None
    expert_counts: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None

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

    def get_assignment_buffers(
        self,
        num_assignments: int,
        *,
        weight_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reusable assignment-sized buffers."""

        token_permutation = self._ensure_1d(
            "token_permutation",
            num_assignments,
            dtype=torch.int64,
            device=device,
        )
        inverse_permutation = self._ensure_1d(
            "inverse_permutation",
            num_assignments,
            dtype=torch.int64,
            device=device,
        )
        packed_expert_ids = self._ensure_1d(
            "packed_expert_ids",
            num_assignments,
            dtype=torch.int64,
            device=device,
        )
        packed_token_indices = self._ensure_1d(
            "packed_token_indices",
            num_assignments,
            dtype=torch.int64,
            device=device,
        )
        packed_routing_weights = self._ensure_1d(
            "packed_routing_weights",
            num_assignments,
            dtype=weight_dtype,
            device=device,
        )
        return (
            token_permutation,
            inverse_permutation,
            packed_expert_ids,
            packed_token_indices,
            packed_routing_weights,
        )

    def get_expert_buffers(
        self,
        num_experts: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable expert-sized buffers."""

        expert_counts = self._ensure_1d(
            "expert_counts",
            num_experts,
            dtype=torch.int64,
            device=device,
        )
        expert_offsets = self._ensure_1d(
            "expert_offsets",
            num_experts + 1,
            dtype=torch.int64,
            device=device,
        )
        return expert_counts, expert_offsets

    def estimated_bytes(self) -> int:
        """Estimate total workspace buffer size in bytes."""

        total = 0
        for tensor in (
            self.token_permutation,
            self.inverse_permutation,
            self.packed_expert_ids,
            self.packed_token_indices,
            self.packed_routing_weights,
            self.expert_counts,
            self.expert_offsets,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
