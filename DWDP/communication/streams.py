"""CUDA stream ownership for the DWDP communication runtime.

Streams are intentionally managed separately from the communication planner:
the planner decides *what* should happen, while this module owns CUDA objects
that later phases use to perform it.  CUDA events (Phase 1 subsystem 2) are
required before work is submitted across these streams.
"""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import ContextManager

import torch


class StreamRole(str, Enum):
    """Named CUDA streams owned by one DWDP runtime instance."""

    COPY = "copy"
    COMPUTE = "compute"


class CudaStreams:
    """Lazily allocate and own one copy stream and one compute stream.

    A stream pair is bound to exactly one CUDA device for its lifetime.  CPU
    execution remains a supported reference path: :meth:`ensure` is then a
    no-op and :meth:`use` returns a no-op context manager.  This makes runtime
    integration device agnostic without pretending that CPU has CUDA streams.

    The class intentionally does not submit work or synchronize streams.  The
    event layer will establish the copy-to-compute dependencies required for
    safe overlap in the next subsystem.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._device: torch.device | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._compute_stream: torch.cuda.Stream | None = None

    @property
    def enabled(self) -> bool:
        """Whether CUDA stream creation is enabled for this runtime."""

        return self._enabled

    @property
    def device(self) -> torch.device | None:
        """CUDA device bound to this stream pair, if any."""

        return self._device

    @property
    def initialized(self) -> bool:
        """Whether the CUDA stream pair has been allocated."""

        return self._copy_stream is not None

    def ensure(self, device: torch.device | str) -> bool:
        """Allocate the stream pair for ``device`` on first CUDA use.

        Returns ``True`` only when a CUDA pair is available.  A runtime cannot
        silently migrate stream-owned work between GPUs, so rebinding to a
        different CUDA device is rejected.
        """

        resolved = torch.device(device)
        if not self._enabled or resolved.type != "cuda":
            return False
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA streams requested for a CUDA device, but CUDA is unavailable"
            )

        # Explicitly resolve the index so cuda and cuda:0 compare equally.
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        resolved = torch.device("cuda", index)
        if self._device is not None and self._device != resolved:
            raise RuntimeError(
                "A DWDP runtime owns streams for one CUDA device; "
                f"already bound to {self._device}, cannot bind to {resolved}"
            )
        if self.initialized:
            return True

        self._device = resolved
        self._copy_stream = torch.cuda.Stream(device=resolved)
        self._compute_stream = torch.cuda.Stream(device=resolved)
        return True

    def get(self, role: StreamRole) -> torch.cuda.Stream:
        """Return a previously initialized CUDA stream for ``role``."""

        stream = self._copy_stream if role is StreamRole.COPY else self._compute_stream
        if stream is None:
            raise RuntimeError(
                "CUDA streams are not initialized; call ensure(cuda_device) first"
            )
        return stream

    @property
    def copy(self) -> torch.cuda.Stream:
        """Dedicated stream for future host/device and peer copies."""

        return self.get(StreamRole.COPY)

    @property
    def compute(self) -> torch.cuda.Stream:
        """Dedicated stream for future expert compute."""

        return self.get(StreamRole.COMPUTE)

    def use(self, role: StreamRole, device: torch.device | str) -> ContextManager[None]:
        """Return a stream context for CUDA, or a no-op context for CPU.

        This API is exposed now for later execution backends.  The reference
        executor does not use it yet because cross-stream event dependencies
        are not installed until the following subsystem.
        """

        if not self.ensure(device):
            return nullcontext()
        return torch.cuda.stream(self.get(role))
