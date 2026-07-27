"""Runtime communication primitives independent of planning policy.

This package owns the execution-side communication foundations.  It is kept
separate from :mod:`DWDP.comms_planner`, which only describes work.
"""

from .execution import ExecutionCommunicationEngine, ExpertPointer
from .streams import CudaStreams, StreamRole

__all__ = ["CudaStreams", "ExecutionCommunicationEngine", "ExpertPointer", "StreamRole"]
