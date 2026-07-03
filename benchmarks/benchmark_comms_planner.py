from __future__ import annotations

import argparse
import time

import torch

from DWDP.comms_planner import (
    CommunicationPlannerConfig,
    CommunicationPlannerWorkspace,
    StaticCommunicationPlanner,
)
from DWDP.comms_planner.utils import estimate_tensor_bytes
from DWDP.scheduler import (
    DependencyMetadata as SchedulerDependencyMetadata,
    ExecutionPlan,
    SchedulerStatistics,
    SynchronizationMetadata as SchedulerSynchronizationMetadata,
)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_execution_plan(*, active_experts: int, device: str) -> ExecutionPlan:
    expert_queue = torch.arange(active_experts, dtype=torch.int64, device=device)
    starts = torch.arange(active_experts, dtype=torch.int64, device=device) * 8
    ends = starts + 8
    order = torch.arange(active_experts, dtype=torch.int64, device=device)
    return ExecutionPlan(
        execution_order=order,
        expert_queue=expert_queue,
        expert_starts=starts,
        expert_ends=ends,
        expert_counts=torch.full((active_experts,), 8, dtype=torch.int64, device=device),
        execution_priority=order,
        stream_assignments=torch.zeros(active_experts, dtype=torch.int64, device=device),
        batches=(),
        synchronization=SchedulerSynchronizationMetadata(
            barrier_after_batch=torch.zeros(active_experts, dtype=torch.bool, device=device),
        ),
        dependencies=SchedulerDependencyMetadata(
            dependency_src=torch.empty(0, dtype=torch.int64, device=device),
            dependency_dst=torch.empty(0, dtype=torch.int64, device=device),
        ),
        statistics=SchedulerStatistics(
            num_experts=active_experts,
            num_active_experts=active_experts,
            num_empty_experts=0,
            num_execution_batches=active_experts,
            num_assignments=active_experts * 8,
            max_tokens_per_expert=8 if active_experts else 0,
            min_tokens_per_active_expert=8 if active_experts else 0,
            scheduling_policy="round_robin",
        ),
        scheduling_policy="round_robin",
        deterministic=True,
    )


def time_callable(fn, *, warmup: int, iters: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the DWDP Comms Planner.")
    parser.add_argument("--active-experts", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    execution_plan = make_execution_plan(
        active_experts=args.active_experts,
        device=args.device,
    )
    planner = StaticCommunicationPlanner(
        CommunicationPlannerConfig(world_size=args.world_size)
    ).eval()
    planner_no_workspace = StaticCommunicationPlanner(
        CommunicationPlannerConfig(world_size=args.world_size, enable_workspace=False)
    ).eval()
    workspace = CommunicationPlannerWorkspace()

    def run_with_workspace():
        return planner(execution_plan, workspace=workspace)

    def run_without_workspace():
        return planner_no_workspace(execution_plan, workspace=workspace)

    def run_graph_generation():
        plan = planner(execution_plan, workspace=workspace)
        return plan.graph

    def run_cost_model_generation():
        plan = planner(execution_plan, workspace=workspace)
        return plan.cost_model

    def run_statistics_generation():
        plan = planner(execution_plan, workspace=workspace)
        return plan.statistics

    with torch.no_grad():
        with_workspace_seconds = time_callable(
            run_with_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        without_workspace_seconds = time_callable(
            run_without_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        graph_seconds = time_callable(
            run_graph_generation,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        cost_seconds = time_callable(
            run_cost_model_generation,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        stats_seconds = time_callable(
            run_statistics_generation,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        plan = run_with_workspace()

    metadata_bytes = (
        estimate_tensor_bytes(plan.local_expert_ids)
        + estimate_tensor_bytes(plan.remote_expert_ids)
        + estimate_tensor_bytes(plan.graph.node_ids)
        + estimate_tensor_bytes(plan.graph.edge_src)
        + estimate_tensor_bytes(plan.graph.edge_dst)
        + estimate_tensor_bytes(plan.topology.gpu_ids)
        + estimate_tensor_bytes(plan.topology.numa_domains)
        + estimate_tensor_bytes(plan.dependencies.dependency_src)
        + estimate_tensor_bytes(plan.dependencies.dependency_dst)
    )

    print(f"device={args.device}")
    print("planner_policy=static")
    print(f"active_experts={args.active_experts} world_size={args.world_size}")
    print(f"planner_latency_with_workspace_us={with_workspace_seconds * 1e6:.2f}")
    print(f"planner_latency_without_workspace_us={without_workspace_seconds * 1e6:.2f}")
    print(f"communication_graph_generation_us={graph_seconds * 1e6:.2f}")
    print(f"cost_model_generation_us={cost_seconds * 1e6:.2f}")
    print(f"statistics_generation_us={stats_seconds * 1e6:.2f}")
    print(f"single_gpu_remote_transfers={len(plan.transfer_descriptors)}")
    print(f"metadata_bytes={metadata_bytes}")
    print(f"workspace_bytes={workspace.estimated_bytes()}")


if __name__ == "__main__":
    main()
