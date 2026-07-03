from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CommunicationCostEstimate:
    """Cost estimate for one future communication descriptor."""

    descriptor_id: int
    estimated_bytes: int
    estimated_latency_us: float
    estimated_bandwidth_gbps: float
    communication_priority: int
    critical_path_us: float
    transfer_duration_us: float
    prefetch_window_us: float
    overlap_estimate_us: float


@dataclass(slots=True)
class CommunicationCostModel:
    """Aggregate communication cost model placeholder."""

    estimates: tuple[CommunicationCostEstimate, ...]
    total_estimated_bytes: int
    total_estimated_latency_us: float
    critical_path_us: float
    estimated_bandwidth_gbps: float
