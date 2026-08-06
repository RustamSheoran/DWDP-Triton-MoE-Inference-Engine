from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MergerWorkspace:
    """Compatibility placeholder for merger backends.

    Final token-major outputs are forward-owned tensors. Reusing one as a
    workspace would let a later MoE layer overwrite an earlier layer's input.
    """

    def estimated_bytes(self) -> int:
        """Estimate allocated workspace bytes."""

        return 0
