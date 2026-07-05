from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .metrics import RuntimeStatistics


def _to_dict(value: Any) -> dict[str, object]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def extract_runtime_statistics(pipeline_output: Any, workspace_bytes: int | None = None) -> RuntimeStatistics:
    """Extract runtime statistics from a `RuntimePipelineOutput`-like object."""

    dispatch_plan = getattr(pipeline_output, "dispatch_plan", None)
    execution_plan = getattr(pipeline_output, "execution_plan", None)
    communication_plan = getattr(pipeline_output, "communication_plan", None)
    executor_output = getattr(pipeline_output, "executor_output", None)
    merger_output = getattr(pipeline_output, "merger_output", None)
    router_output = getattr(pipeline_output, "router_output", None)

    routing_distribution: dict[str, int] = {}
    metadata = getattr(router_output, "metadata", None)
    expert_counts = getattr(metadata, "expert_counts", None)
    if expert_counts is not None:
        try:
            routing_distribution = {
                str(index): int(value)
                for index, value in enumerate(expert_counts.detach().cpu().tolist())
            }
        except Exception:
            routing_distribution = {}

    scheduler_stats = getattr(execution_plan, "statistics", None)
    num_experts = getattr(scheduler_stats, "num_experts", None)
    active_experts = getattr(scheduler_stats, "num_active_experts", None)
    workspace_allocations = {}
    if workspace_bytes is not None:
        workspace_allocations["total_bytes"] = workspace_bytes

    return RuntimeStatistics(
        workspace_allocations=workspace_allocations,
        workspace_reuse={},
        num_experts=num_experts,
        active_experts=active_experts,
        routing_distribution=routing_distribution,
        dispatcher_statistics=_to_dict(getattr(dispatch_plan, "metadata", None)),
        scheduler_statistics=_to_dict(scheduler_stats),
        communication_planner_statistics=_to_dict(getattr(communication_plan, "statistics", None)),
        executor_statistics=_to_dict(getattr(executor_output, "statistics", None)),
        merger_statistics=_to_dict(getattr(merger_output, "statistics", None)),
    )
