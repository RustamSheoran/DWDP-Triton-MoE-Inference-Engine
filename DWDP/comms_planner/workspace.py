from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CommunicationPlannerWorkspace:
    """Reusable buffers for communication planning metadata."""

    empty_int64: torch.Tensor | None = None
    empty_float32: torch.Tensor | None = None
    empty_bool: torch.Tensor | None = None
    local_gpu_ids: torch.Tensor | None = None
    numa_domains: torch.Tensor | None = None

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

    def get_empty_int64(self, *, device: torch.device) -> torch.Tensor:
        """Return a reusable empty int64 tensor."""

        return self._ensure_1d("empty_int64", 0, dtype=torch.int64, device=device)

    def get_empty_float32(self, *, device: torch.device) -> torch.Tensor:
        """Return a reusable empty float32 tensor."""

        return self._ensure_1d("empty_float32", 0, dtype=torch.float32, device=device)

    def get_empty_bool(self, *, device: torch.device) -> torch.Tensor:
        """Return a reusable empty bool tensor."""

        return self._ensure_1d("empty_bool", 0, dtype=torch.bool, device=device)

    def get_topology_buffers(
        self,
        world_size: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable topology tensors."""

        gpu_ids = self._ensure_1d(
            "local_gpu_ids",
            world_size,
            dtype=torch.int64,
            device=device,
        )
        numa_domains = self._ensure_1d(
            "numa_domains",
            world_size,
            dtype=torch.int64,
            device=device,
        )
        return gpu_ids, numa_domains

    def estimated_bytes(self) -> int:
        """Estimate allocated workspace bytes."""

        total = 0
        for tensor in (
            self.empty_int64,
            self.empty_float32,
            self.empty_bool,
            self.local_gpu_ids,
            self.numa_domains,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
