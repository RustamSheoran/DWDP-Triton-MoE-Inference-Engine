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
        # Local experts execute through their original module storage. Native
        # staging is initialized explicitly by distributed bootstrap, where
        # remote IPC experts actually require it. Initializing it here reserved
        # two 16 MiB CUDA buffers per MoE layer on single-GPU inference.
        pointer = ExpertPointer(module=module, device_pointers=pointers)
        self._expert_pointers[expert_id] = pointer
        return pointer

    def getResidentPointer(self, expert_id: int) -> ExpertPointer:
        return self.getWeight(expert_id)

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._registry.expert_ids

    def prefetch(self, expert_id: int) -> None:
        """Prefetch expert weights via C++ engine if available."""
        if self._native_engine is not None:
            self._native_engine.prefetch(expert_id)

    def wait(self, expert_id: int) -> None:
        """Wait for expert weights via C++ engine if available."""
        if self._native_engine is not None:
            self._native_engine.wait(expert_id)

    def swap_buffers(self) -> None:
        """Swap double buffers in C++ engine if available."""
        if self._native_engine is not None:
            self._native_engine.swap_buffers()

    def is_native_available(self) -> bool:
        """Return True if native C++ engine is active."""
        return self._native_engine is not None

    def ensure_native(self, device: torch.device | None = None) -> bool:
        """Eagerly initialize the native engine before the first forward pass.

        Distributed startup must register remote IPC experts at load time
        (Project 1 spec), which happens before any ``getWeight`` call would
        have lazily constructed the engine.
        """

        if device is None:
            for expert_id in self._registry.expert_ids:
                for parameter in self._registry.get(expert_id).parameters(recurse=True):
                    if parameter.device.type == "cuda":
                        device = parameter.device
                        break
                if device is not None:
                    break
        if device is None:
            return False
        return self._ensure_native_engine(device)

    def register_remote_expert(
        self, expert_id: int, ipc_handle: bytes, size_bytes: int
    ) -> bool:
        """Register a peer rank's expert weights through its CUDA IPC handle.

        Returns True when the C++ engine accepted the handle. Remote experts
        registered this way become prefetchable over NVLink P2P.
        """

        if self._native_engine is None or not ipc_handle:
            return False
        try:
            self._native_engine.register_ipc_expert(expert_id, ipc_handle, size_bytes)
            self._native_registered.add(expert_id)
            return True
        except Exception:
            logger.warning(
                "failed to register remote expert %d via IPC", expert_id, exc_info=True
            )
            return False


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
