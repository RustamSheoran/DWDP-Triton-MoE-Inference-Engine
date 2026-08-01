"""PyTorch runtime bridge for the compiled DWDP communication engine ABI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from DWDP.executor.experts import ExpertRegistry

logger = logging.getLogger(__name__)

# Attempt to load the compiled C++/CUDA communication engine.
# Falls back to pure-Python weight resolution when unavailable.
try:
    import dwdp_communication_ext as _native  # type: ignore[import-not-found]

    _NATIVE_AVAILABLE = True
except ImportError:
    _native = None
    _NATIVE_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class ExpertPointer:
    """Executable expert reference resolved only through the communication layer."""

    module: nn.Module
    device_pointers: tuple[int, ...]


# Default staging buffer size (16 MB per buffer, 32 MB total for double-buffer).
# Sufficient for single-GPU local expert registration where no actual staging
# copies occur.  Multi-GPU deployments should configure this explicitly.
_DEFAULT_STAGING_BYTES = 16 * 1024 * 1024


class ExecutionCommunicationEngine(nn.Module):
    """Communication-engine boundary used by the Python reference executor.

    When the compiled C++/CUDA native engine (``dwdp_communication_ext``) is
    available and the runtime is on CUDA, this class delegates weight
    registration and pointer resolution to the C++ ``CommunicationEngine``.
    The C++ engine manages CUDA IPC handles, double-buffered staging areas,
    asynchronous prefetch workers, and stream synchronization as specified
    by the Project 1 architecture.

    When the native engine is unavailable (CPU-only environments, missing
    CUDA toolkit, or extension not compiled), the class falls back to the
    original pure-Python behavior: caching local ``nn.Module`` references
    and returning their device pointers directly.
    """

    def __init__(self, experts: ExpertRegistry) -> None:
        super().__init__()
        self._registry = experts
        self._expert_pointers: dict[int, ExpertPointer] = {}
        self._native_engine = None
        self._native_registered: set[int] = set()

    def _ensure_native_engine(self, device: torch.device) -> bool:
        """Lazily initialize the C++ CommunicationEngine on first CUDA access."""
        if self._native_engine is not None:
            return True
        if not _NATIVE_AVAILABLE or device.type != "cuda":
            return False
        try:
            device_id = device.index if device.index is not None else 0
            engine = _native.CommunicationEngine(device_id)
            engine.initialize(_DEFAULT_STAGING_BYTES)
            self._native_engine = engine
            logger.info(
                "C++ CommunicationEngine initialized on device %d", device_id
            )
            return True
        except Exception:
            logger.warning(
                "Failed to initialize C++ CommunicationEngine; "
                "falling back to pure-Python weight resolution",
                exc_info=True,
            )
            return False

    def _register_expert_native(self, expert_id: int, module: nn.Module) -> None:
        """Register all CUDA parameters of an expert with the C++ engine."""
        if expert_id in self._native_registered:
            return
        for parameter in module.parameters(recurse=True):
            if parameter.device.type == "cuda":
                ptr = parameter.data_ptr()
                size = parameter.nelement() * parameter.element_size()
                self._native_engine.register_expert(expert_id, ptr, size)
                self._native_registered.add(expert_id)
                break  # Register with the first CUDA parameter's address.

    def getWeight(self, expert_id: int) -> ExpertPointer:
        """Resolve an expert once and reuse its immutable module reference.

        The reference execution path calls this from the per-expert hot loop.
        Enumerating every parameter to rebuild debug pointer metadata on each
        invocation is pure Python overhead and does not affect execution.
        """

        cached = self._expert_pointers.get(expert_id)
        if cached is not None:
            return cached
        module = self._registry.get(expert_id)
        pointers = tuple(
            parameter.data_ptr()
            for parameter in module.parameters(recurse=True)
            if parameter.device.type == "cuda"
        )
        # Register with the C++ engine if available.
        if pointers:
            device = next(
                p.device
                for p in module.parameters(recurse=True)
                if p.device.type == "cuda"
            )
            if self._ensure_native_engine(device):
                self._register_expert_native(expert_id, module)

        pointer = ExpertPointer(module=module, device_pointers=pointers)
        self._expert_pointers[expert_id] = pointer
        return pointer

    def getResidentPointer(self, expert_id: int) -> ExpertPointer:
        return self.getWeight(expert_id)

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._registry.expert_ids

    def shutdown(self) -> None:
        """Shutdown the C++ engine and release all IPC handles."""
        if self._native_engine is not None:
            self._native_engine.shutdown()
            self._native_engine = None
            self._native_registered.clear()
            logger.info("C++ CommunicationEngine shut down")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
