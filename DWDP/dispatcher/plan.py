from __future__ import annotations

from dataclasses import dataclass

from .assignments import ExpertAssignments
from .metadata import DispatchMetadata


@dataclass(slots=True)
class DispatchPlan:
    """Structured dispatch plan returned by the dispatcher."""

    assignments: ExpertAssignments
    metadata: DispatchMetadata
