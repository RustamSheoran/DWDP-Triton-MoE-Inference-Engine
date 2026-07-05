from __future__ import annotations

import argparse
import time

import torch

from DWDP.executor import (
    ExecutionMetadata,
    ExecutionStatistics,
    ExecutorOutput,
    OutputMetadata,
)
from DWDP.executor.metadata import TimingMetadata as ExecutorTimingMetadata
from DWDP.executor.metadata import WorkspaceMetadata as ExecutorWorkspaceMetadata
from DWDP.merger import MergerConfig, MergerWorkspace, PyTorchMerger
from DWDP.merger.utils import estimate_tensor_bytes


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_executor_output(tokens: int, top_k: int, hidden_size: int, device: str) -> ExecutorOutput:
    num_assignments = tokens * top_k
    packed = torch.randn(num_assignments, hidden_size, device=device)
    weights = torch.rand(num_assignments, device=device)
    weighted = packed * weights.unsqueeze(-1)
    inverse = torch.arange(num_assignments, dtype=torch.int64, device=device)
    return ExecutorOutput(
        packed_expert_outputs=packed,
        weighted_expert_outputs=weighted,
        expert_outputs=(),
        output_metadata=OutputMetadata(
            packed_token_indices=torch.arange(num_assignments, dtype=torch.int64, device=device) // top_k,
            packed_expert_ids=torch.zeros(num_assignments, dtype=torch.int64, device=device),
            packed_routing_weights=weights,
            token_permutation=inverse,
            inverse_permutation=inverse,
            token_shape=(tokens,),
            top_k=top_k,
        ),
        execution_metadata=ExecutionMetadata(
            execution_order=torch.empty(0, dtype=torch.int64, device=device),
            expert_queue=torch.empty(0, dtype=torch.int64, device=device),
            expert_starts=torch.empty(0, dtype=torch.int64, device=device),
            expert_ends=torch.empty(0, dtype=torch.int64, device=device),
            stream_assignments=torch.empty(0, dtype=torch.int64, device=device),
            communication_remote_expert_ids=torch.empty(0, dtype=torch.int64, device=device),
            communication_policy="static",
            scheduling_policy="round_robin",
        ),
        statistics=ExecutionStatistics(0, 0, tokens, num_assignments, hidden_size, hidden_size, "benchmark"),
        timing=ExecutorTimingMetadata(),
        workspace=ExecutorWorkspaceMetadata(False, 0),
        backend="benchmark",
        deterministic=True,
    )


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
    parser = argparse.ArgumentParser(description="Benchmark the DWDP PyTorch merger.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    executor_output = make_executor_output(args.tokens, args.top_k, args.hidden_size, args.device)
    merger = PyTorchMerger(MergerConfig()).eval()
    merger_no_workspace = PyTorchMerger(MergerConfig(enable_workspace=False)).eval()
    workspace = MergerWorkspace()

    def run_with_workspace():
        return merger(executor_output, workspace=workspace)

    def run_without_workspace():
        return merger_no_workspace(executor_output, workspace=workspace)

    with torch.no_grad():
        with_workspace_seconds = time_callable(run_with_workspace, args.warmup, args.iters, args.device)
        without_workspace_seconds = time_callable(run_without_workspace, args.warmup, args.iters, args.device)
        output = run_with_workspace()

    throughput = args.tokens / with_workspace_seconds
    print(f"device={args.device}")
    print("backend=pytorch")
    print(f"tokens={args.tokens} top_k={args.top_k} hidden_size={args.hidden_size}")
    print(f"latency_with_workspace_us={with_workspace_seconds * 1e6:.2f}")
    print(f"latency_without_workspace_us={without_workspace_seconds * 1e6:.2f}")
    print(f"tokens_per_second={throughput:.2f}")
    print(f"output_bytes={estimate_tensor_bytes(output.hidden_states)}")
    print(f"workspace_bytes={workspace.estimated_bytes()}")


if __name__ == "__main__":
    main()
