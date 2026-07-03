from __future__ import annotations

import argparse
import time

import torch

from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments
from DWDP.scheduler import RoundRobinScheduler, SchedulerConfig, SchedulerWorkspace
from DWDP.scheduler.utils import estimate_tensor_bytes


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_dispatch_plan(
    *,
    num_experts: int,
    total_assignments: int,
    active_fraction: float,
    device: str,
) -> DispatchPlan:
    active_experts = max(1, min(num_experts, int(num_experts * active_fraction)))
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if active_experts > 0:
        base = total_assignments // active_experts
        remainder = total_assignments % active_experts
        counts[:active_experts] = base
        if remainder > 0:
            counts[:remainder] += 1
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)
    assignments = ExpertAssignments(
        expert_ids=torch.empty(total_assignments, dtype=torch.int64, device=device),
        packed_token_indices=torch.empty(total_assignments, dtype=torch.int64, device=device),
        packed_routing_weights=torch.empty(total_assignments, dtype=torch.float32, device=device),
    )
    metadata = DispatchMetadata(
        num_tokens=total_assignments,
        num_assignments=total_assignments,
        num_experts=num_experts,
        top_k=1,
        token_shape=(total_assignments,),
        expert_counts=counts,
        expert_offsets=offsets,
        token_permutation=torch.empty(total_assignments, dtype=torch.int64, device=device),
        inverse_permutation=torch.empty(total_assignments, dtype=torch.int64, device=device),
        destination_positions=torch.empty(total_assignments, dtype=torch.int64, device=device),
        stable_order=True,
        algorithm="benchmark",
    )
    return DispatchPlan(assignments=assignments, metadata=metadata)


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
    parser = argparse.ArgumentParser(description="Benchmark the DWDP scheduler.")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--assignments", type=int, default=4096)
    parser.add_argument("--active-fraction", type=float, default=1.0)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dispatch_plan = make_dispatch_plan(
        num_experts=args.num_experts,
        total_assignments=args.assignments,
        active_fraction=args.active_fraction,
        device=args.device,
    )
    scheduler = RoundRobinScheduler(
        SchedulerConfig(stream_count=args.stream_count)
    ).eval()
    scheduler_no_workspace = RoundRobinScheduler(
        SchedulerConfig(stream_count=args.stream_count, enable_workspace=False)
    ).eval()
    workspace = SchedulerWorkspace()

    def run_with_workspace():
        return scheduler(dispatch_plan, workspace=workspace)

    def run_without_workspace():
        return scheduler_no_workspace(dispatch_plan, workspace=workspace)

    def run_metadata_only():
        plan = scheduler(dispatch_plan, workspace=workspace)
        return (
            plan.execution_order,
            plan.expert_queue,
            plan.expert_starts,
            plan.expert_ends,
            plan.stream_assignments,
        )

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
        metadata_seconds = time_callable(
            run_metadata_only,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        plan = run_with_workspace()

    active_experts = plan.statistics.num_active_experts
    throughput = active_experts / with_workspace_seconds if with_workspace_seconds > 0 else 0.0
    output_bytes = (
        estimate_tensor_bytes(plan.execution_order)
        + estimate_tensor_bytes(plan.expert_queue)
        + estimate_tensor_bytes(plan.expert_starts)
        + estimate_tensor_bytes(plan.expert_ends)
        + estimate_tensor_bytes(plan.expert_counts)
        + estimate_tensor_bytes(plan.execution_priority)
        + estimate_tensor_bytes(plan.stream_assignments)
        + estimate_tensor_bytes(plan.synchronization.barrier_after_batch)
        + estimate_tensor_bytes(plan.dependencies.dependency_src)
        + estimate_tensor_bytes(plan.dependencies.dependency_dst)
    )

    print(f"device={args.device}")
    print(f"policy=round_robin")
    print(f"experts={args.num_experts} active_experts={active_experts}")
    print(f"assignments={args.assignments} stream_count={args.stream_count}")
    print(f"latency_with_workspace_us={with_workspace_seconds * 1e6:.2f}")
    print(f"latency_without_workspace_us={without_workspace_seconds * 1e6:.2f}")
    print(f"metadata_generation_us={metadata_seconds * 1e6:.2f}")
    print(f"throughput_active_experts_per_second={throughput:.2f}")
    print(f"output_metadata_bytes={output_bytes}")
    print(f"workspace_bytes={workspace.estimated_bytes()}")


if __name__ == "__main__":
    main()
