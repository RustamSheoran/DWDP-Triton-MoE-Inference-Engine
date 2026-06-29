from __future__ import annotations

import argparse
import time

import torch

from DWDP.dispatcher import DispatchWorkspace, DispatcherConfig, ExpertMajorDispatcher
from DWDP.dispatcher.ops import (
    compute_destination_positions,
    compute_expert_histogram,
    exclusive_cumsum,
    invert_permutation,
    pack_routing_weights,
    pack_token_indices,
    stable_expert_permutation,
)
from DWDP.dispatcher.utils import estimate_tensor_bytes
from DWDP.router import RouterOutput


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_router_output(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    weight_dtype: torch.dtype,
    device: str,
) -> RouterOutput:
    topk_indices = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, top_k),
        device=device,
        dtype=torch.int64,
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        device=device,
        dtype=weight_dtype,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return RouterOutput(
        router_logits=torch.empty(0, device=device),
        routing_probabilities=torch.empty(0, device=device),
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        metadata=None,
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
    parser = argparse.ArgumentParser(description="Benchmark the DWDP dispatcher.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.dtype]

    router_output = make_router_output(
        num_tokens=args.tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        weight_dtype=weight_dtype,
        device=args.device,
    )
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=args.num_experts)).eval()
    sort_dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=args.num_experts, algorithm="stable_sort")
    ).eval()
    workspace = DispatchWorkspace()
    sort_workspace = DispatchWorkspace()

    flat_expert_indices = router_output.topk_indices.reshape(-1)
    flat_routing_weights = router_output.topk_weights.reshape(-1)
    counts = compute_expert_histogram(flat_expert_indices, args.num_experts)
    offsets = exclusive_cumsum(counts)

    def run_scatter_no_workspace():
        return dispatcher(router_output)

    def run_scatter_with_workspace():
        return dispatcher(router_output, workspace=workspace)

    def run_sort_no_workspace():
        return sort_dispatcher(router_output)

    def run_sort_with_workspace():
        return sort_dispatcher(router_output, workspace=sort_workspace)

    def run_histogram():
        return compute_expert_histogram(flat_expert_indices, args.num_experts)

    def run_prefix_sum():
        histogram = compute_expert_histogram(flat_expert_indices, args.num_experts)
        return exclusive_cumsum(histogram)

    def run_destination_positions():
        return compute_destination_positions(flat_expert_indices, offsets)

    def run_sort_permutation():
        return stable_expert_permutation(flat_expert_indices)

    def run_inverse():
        _, permutation = stable_expert_permutation(flat_expert_indices)
        return invert_permutation(permutation)

    def run_pack():
        _, permutation = stable_expert_permutation(flat_expert_indices)
        packed_token_indices = pack_token_indices(permutation, args.top_k)
        packed_routing_weights = pack_routing_weights(flat_routing_weights, permutation)
        return packed_token_indices, packed_routing_weights

    with torch.no_grad():
        scatter_no_workspace_seconds = time_callable(
            run_scatter_no_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        scatter_with_workspace_seconds = time_callable(
            run_scatter_with_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        sort_no_workspace_seconds = time_callable(
            run_sort_no_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        sort_with_workspace_seconds = time_callable(
            run_sort_with_workspace,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        histogram_seconds = time_callable(
            run_histogram,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        prefix_seconds = time_callable(
            run_prefix_sum,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        destination_seconds = time_callable(
            run_destination_positions,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        permutation_seconds = time_callable(
            run_sort_permutation,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        inverse_seconds = time_callable(
            run_inverse,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        pack_seconds = time_callable(
            run_pack,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        scatter_plan = run_scatter_with_workspace()
        sort_plan = run_sort_with_workspace()

    if not torch.equal(scatter_plan.assignments.expert_ids, sort_plan.assignments.expert_ids):
        raise RuntimeError("counting_scatter and stable_sort produced different expert ids")
    if not torch.equal(
        scatter_plan.assignments.packed_token_indices,
        sort_plan.assignments.packed_token_indices,
    ):
        raise RuntimeError("counting_scatter and stable_sort produced different packed token indices")
    if not torch.allclose(
        scatter_plan.assignments.packed_routing_weights,
        sort_plan.assignments.packed_routing_weights,
    ):
        raise RuntimeError("counting_scatter and stable_sort produced different packed routing weights")

    num_assignments = args.tokens * args.top_k
    scatter_throughput = num_assignments / scatter_with_workspace_seconds
    sort_throughput = num_assignments / sort_with_workspace_seconds
    scatter_packed_bytes = (
        estimate_tensor_bytes(scatter_plan.assignments.expert_ids)
        + estimate_tensor_bytes(scatter_plan.assignments.packed_token_indices)
        + estimate_tensor_bytes(scatter_plan.assignments.packed_routing_weights)
    )
    sort_packed_bytes = (
        estimate_tensor_bytes(sort_plan.assignments.expert_ids)
        + estimate_tensor_bytes(sort_plan.assignments.packed_token_indices)
        + estimate_tensor_bytes(sort_plan.assignments.packed_routing_weights)
    )

    print(f"device={args.device} dtype={args.dtype}")
    print(f"tokens={args.tokens} experts={args.num_experts} top_k={args.top_k}")
    print(f"assignments={num_assignments}")
    print(f"counting_scatter_latency_no_workspace_us={scatter_no_workspace_seconds * 1e6:.2f}")
    print(f"counting_scatter_latency_with_workspace_us={scatter_with_workspace_seconds * 1e6:.2f}")
    print(f"stable_sort_latency_no_workspace_us={sort_no_workspace_seconds * 1e6:.2f}")
    print(f"stable_sort_latency_with_workspace_us={sort_with_workspace_seconds * 1e6:.2f}")
    print(f"counting_scatter_throughput_assignments_per_second={scatter_throughput:.2f}")
    print(f"stable_sort_throughput_assignments_per_second={sort_throughput:.2f}")
    print(f"counting_scatter_vs_sort_speedup={sort_with_workspace_seconds / scatter_with_workspace_seconds:.4f}")
    print(f"histogram_us={histogram_seconds * 1e6:.2f}")
    print(f"prefix_sum_us={prefix_seconds * 1e6:.2f}")
    print(f"destination_positions_us={destination_seconds * 1e6:.2f}")
    print(f"stable_sort_permutation_us={permutation_seconds * 1e6:.2f}")
    print(f"inverse_permutation_us={inverse_seconds * 1e6:.2f}")
    print(f"packing_us={pack_seconds * 1e6:.2f}")
    print(f"counting_scatter_packed_output_bytes={scatter_packed_bytes}")
    print(f"stable_sort_packed_output_bytes={sort_packed_bytes}")
    print(f"counting_scatter_workspace_bytes={workspace.estimated_bytes()}")
    print(f"stable_sort_workspace_bytes={sort_workspace.estimated_bytes()}")


if __name__ == "__main__":
    main()
