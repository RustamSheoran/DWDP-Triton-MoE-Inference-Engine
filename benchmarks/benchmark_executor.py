from __future__ import annotations

import argparse
import time

import torch

from DWDP.comms_planner import (
    CommunicationCostModel,
    CommunicationGraph,
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata as CommunicationDependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata as CommunicationSynchronizationMetadata,
    TopologyMetadata,
)
from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments
from DWDP.executor import ExecutorConfig, ExecutorWorkspace, PyTorchExecutor
from DWDP.executor.experts import ExpertRegistry
from DWDP.executor.utils import estimate_tensor_bytes
from DWDP.scheduler import (
    DependencyMetadata as SchedulerDependencyMetadata,
    ExecutionPlan,
    SchedulerStatistics,
    SynchronizationMetadata as SchedulerSynchronizationMetadata,
)


class MLPExpert(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, intermediate_size),
            torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_plans(num_tokens: int, num_experts: int, device: str) -> tuple[DispatchPlan, ExecutionPlan, CommunicationPlan]:
    counts = torch.full((num_experts,), num_tokens // num_experts, dtype=torch.int64, device=device)
    counts[: num_tokens % num_experts] += 1
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, dtype=torch.int64, device=device), counts)
    token_indices = torch.arange(num_tokens, dtype=torch.int64, device=device)
    routing_weights = torch.ones(num_tokens, dtype=torch.float32, device=device)
    dispatch_plan = DispatchPlan(
        assignments=ExpertAssignments(
            expert_ids=expert_ids,
            packed_token_indices=token_indices,
            packed_routing_weights=routing_weights,
        ),
        metadata=DispatchMetadata(
            num_tokens=num_tokens,
            num_assignments=num_tokens,
            num_experts=num_experts,
            top_k=1,
            token_shape=(num_tokens,),
            expert_counts=counts,
            expert_offsets=offsets,
            token_permutation=torch.arange(num_tokens, dtype=torch.int64, device=device),
            inverse_permutation=torch.arange(num_tokens, dtype=torch.int64, device=device),
            destination_positions=torch.arange(num_tokens, dtype=torch.int64, device=device),
            stable_order=True,
            algorithm="benchmark",
        ),
    )
    active = torch.nonzero(counts > 0, as_tuple=False).flatten()
    order = torch.arange(active.numel(), dtype=torch.int64, device=device)
    execution_plan = ExecutionPlan(
        execution_order=order,
        expert_queue=active,
        expert_starts=offsets.index_select(0, active),
        expert_ends=offsets.index_select(0, active + 1),
        expert_counts=counts.index_select(0, active),
        execution_priority=order,
        stream_assignments=torch.zeros(active.numel(), dtype=torch.int64, device=device),
        batches=(),
        synchronization=SchedulerSynchronizationMetadata(torch.zeros(active.numel(), dtype=torch.bool, device=device)),
        dependencies=SchedulerDependencyMetadata(torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device)),
        statistics=SchedulerStatistics(num_experts, active.numel(), num_experts - active.numel(), active.numel(), num_tokens, int(counts.max().item()), int(counts[counts > 0].min().item()), "round_robin"),
        scheduling_policy="round_robin",
        deterministic=True,
    )
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    communication_plan = CommunicationPlan(
        local_expert_ids=active,
        remote_expert_ids=empty_i64,
        graph=CommunicationGraph((), (), empty_i64, empty_i64, empty_i64),
        communication_descriptors=(),
        transfer_descriptors=(),
        communication_groups=(),
        topology=TopologyMetadata(0, 1, 0, torch.tensor([0], dtype=torch.int64, device=device), torch.tensor([0], dtype=torch.int64, device=device), None, None, None, (), ((0,),), "single_gpu", 0.0, 0.0),
        synchronization=CommunicationSynchronizationMetadata(empty_i64, empty_i64, empty_i64, empty_i64),
        dependencies=CommunicationDependencyMetadata(empty_i64, empty_i64, empty_i64),
        prefetch=PrefetchPlan(empty_i64, empty_i64, empty_f32),
        overlap=OverlapPlan(empty_i64, empty_i64, empty_f32),
        cost_model=CommunicationCostModel((), 0, 0.0, 0.0, 0.0),
        statistics=CommunicationStatistics(active.numel(), 0, 0, 0, 0, 0, 0, 0.0, "static"),
        planner_policy="static",
        deterministic=True,
    )
    return dispatch_plan, execution_plan, communication_plan


def time_callable(fn, warmup: int, iters: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the DWDP PyTorch executor.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    hidden_states = torch.randn(args.tokens, args.hidden_size, device=args.device)
    dispatch_plan, execution_plan, communication_plan = make_plans(args.tokens, args.num_experts, args.device)
    experts = ExpertRegistry([MLPExpert(args.hidden_size, args.intermediate_size).to(args.device) for _ in range(args.num_experts)])
    executor = PyTorchExecutor(ExecutorConfig(), experts).eval()
    workspace = ExecutorWorkspace()

    def run_with_workspace():
        return executor(hidden_states, dispatch_plan, execution_plan, communication_plan, workspace=workspace)

    def run_without_workspace():
        return executor(hidden_states, dispatch_plan, execution_plan, communication_plan)

    with torch.no_grad():
        with_workspace_seconds = time_callable(run_with_workspace, args.warmup, args.iters, args.device)
        without_workspace_seconds = time_callable(run_without_workspace, args.warmup, args.iters, args.device)
        output = run_with_workspace()

    tokens_per_second = args.tokens / with_workspace_seconds
    output_bytes = estimate_tensor_bytes(output.weighted_expert_outputs) + estimate_tensor_bytes(output.packed_expert_outputs)
    print(f"device={args.device}")
    print("backend=pytorch")
    print(f"tokens={args.tokens} hidden_size={args.hidden_size} experts={args.num_experts}")
    print(f"latency_with_workspace_us={with_workspace_seconds * 1e6:.2f}")
    print(f"latency_without_workspace_us={without_workspace_seconds * 1e6:.2f}")
    print(f"tokens_per_second={tokens_per_second:.2f}")
    print(f"executed_experts={output.statistics.num_executed_experts}")
    print(f"output_bytes={output_bytes}")
    print(f"workspace_bytes={workspace.estimated_bytes()}")


if __name__ == "__main__":
    main()
