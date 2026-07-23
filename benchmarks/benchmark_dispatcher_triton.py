"""Compare the reference and tiled Triton dispatcher implementations.

The benchmark is intentionally not executed during package tests. Run it on a
CUDA system, for example:

    python benchmarks/benchmark_dispatcher_triton.py --tokens 4096,16384 --experts 8,64,128 --top-k 2,4,8
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from DWDP.dispatcher import DispatchWorkspace, DispatcherConfig, ExpertMajorDispatcher
from DWDP.router import RouterOutput


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of positive integers")
    return values


def make_router_output(tokens: int, top_k: int, num_experts: int, device: str) -> RouterOutput:
    topk_indices = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int64, device=device)
    topk_weights = torch.rand((tokens, top_k), dtype=torch.float32, device=device)
    return RouterOutput(
        router_logits=torch.empty(0, device=device),
        routing_probabilities=torch.empty(0, device=device),
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        metadata=None,
    )


def _time_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def _peak_memory(fn: Callable[[], object]) -> int:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _kernel_count(fn: Callable[[], object]) -> int | None:
    """Return CUDA event count when the installed profiler exposes it."""

    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        fn()
    return sum(event.count for event in profiler.key_averages() if event.device_type.name == "CUDA")


def _assert_parity(reference_plan, triton_plan) -> None:
    assert torch.equal(reference_plan.metadata.expert_counts, triton_plan.metadata.expert_counts)
    assert torch.equal(reference_plan.metadata.expert_offsets, triton_plan.metadata.expert_offsets)
    assert torch.equal(reference_plan.metadata.token_permutation, triton_plan.metadata.token_permutation)
    assert torch.equal(reference_plan.metadata.inverse_permutation, triton_plan.metadata.inverse_permutation)
    assert torch.equal(reference_plan.assignments.expert_ids, triton_plan.assignments.expert_ids)
    assert torch.equal(reference_plan.assignments.packed_token_indices, triton_plan.assignments.packed_token_indices)
    assert torch.equal(reference_plan.assignments.packed_routing_weights, triton_plan.assignments.packed_routing_weights)


def benchmark_case(tokens: int, num_experts: int, top_k: int, warmup: int, iterations: int) -> None:
    router_output = make_router_output(tokens, top_k, num_experts, "cuda")
    reference = ExpertMajorDispatcher(DispatcherConfig(num_experts=num_experts, algorithm="counting_scatter"))
    triton = ExpertMajorDispatcher(DispatcherConfig(num_experts=num_experts, algorithm="triton_counting_scatter"))
    reference_workspace = DispatchWorkspace()
    triton_workspace = DispatchWorkspace()

    def run_reference():
        return reference(router_output, workspace=reference_workspace)

    def run_triton():
        return triton(router_output, workspace=triton_workspace)

    with torch.no_grad():
        reference_plan = run_reference()
        triton_plan = run_triton()
        _assert_parity(reference_plan, triton_plan)
        reference_latency = _time_cuda(run_reference, warmup, iterations)
        triton_latency = _time_cuda(run_triton, warmup, iterations)
        reference_peak = _peak_memory(run_reference)
        triton_peak = _peak_memory(run_triton)
        triton_kernel_count = _kernel_count(run_triton)

    assignments = tokens * top_k
    kernel_count = "unavailable" if triton_kernel_count is None else str(triton_kernel_count)
    print(
        " ".join(
            (
                f"tokens={tokens}",
                f"experts={num_experts}",
                f"top_k={top_k}",
                f"reference_us={reference_latency * 1e6:.2f}",
                f"triton_us={triton_latency * 1e6:.2f}",
                f"speedup={reference_latency / triton_latency:.3f}",
                f"triton_assignments_per_s={assignments / triton_latency:.2f}",
                f"reference_peak_bytes={reference_peak}",
                f"triton_peak_bytes={triton_peak}",
                f"triton_workspace_bytes={triton_workspace.estimated_bytes()}",
                f"triton_kernel_events={kernel_count}",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reference and tiled Triton DWDP dispatchers.")
    parser.add_argument("--tokens", type=_parse_int_list, default=(4096, 16384))
    parser.add_argument("--experts", type=_parse_int_list, default=(8, 16, 64, 128))
    parser.add_argument("--top-k", type=_parse_int_list, default=(2, 4, 8))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    for tokens in args.tokens:
        for num_experts in args.experts:
            for top_k in args.top_k:
                benchmark_case(tokens, num_experts, top_k, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
