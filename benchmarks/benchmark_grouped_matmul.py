"""Benchmark PyTorch and Triton expert-major grouped matrix multiplication.

This benchmark intentionally measures a pre-packed ``[E, O, K]`` weight
tensor. Weight materialization is excluded from timing because it is a
load-time/backend-packing decision, not a grouped GEMM operation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from DWDP.executor.kernels import grouped_matmul, reference_grouped_matmul


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def make_offsets(num_assignments: int, num_experts: int) -> torch.Tensor:
    counts = torch.full((num_experts,), num_assignments // num_experts, device="cuda", dtype=torch.int64)
    counts[: num_assignments % num_experts] += 1
    return torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def kernel_events(fn: Callable[[], torch.Tensor]) -> int | None:
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        fn()
    return sum(event.count for event in profiler.key_averages() if event.device_type.name == "CUDA")


def benchmark_case(
    tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> None:
    assignments = tokens * top_k
    offsets = make_offsets(assignments, num_experts)
    max_tokens_per_expert = (assignments + num_experts - 1) // num_experts
    inputs = torch.randn(assignments, hidden_size, device="cuda", dtype=dtype)
    weights = torch.randn(num_experts, output_size, hidden_size, device="cuda", dtype=dtype)

    def reference() -> torch.Tensor:
        return reference_grouped_matmul(inputs, weights, offsets)

    def triton() -> torch.Tensor:
        return grouped_matmul(
            inputs,
            weights,
            offsets,
            max_tokens_per_expert=max_tokens_per_expert,
        )

    reference_output = reference()
    triton_output = triton()
    tolerance = 3e-2 if dtype == torch.bfloat16 else 2e-2
    if not torch.allclose(reference_output, triton_output, rtol=tolerance, atol=tolerance):
        raise RuntimeError("Triton grouped matmul failed reference parity")

    reference_latency = time_cuda(reference, warmup, iterations)
    torch.cuda.reset_peak_memory_stats()
    triton_latency = time_cuda(triton, warmup, iterations)
    peak_memory = torch.cuda.max_memory_allocated()
    event_count = kernel_events(triton)
    kernel_count = "unavailable" if event_count is None else str(event_count)
    print(
        " ".join(
            (
                f"tokens={tokens}",
                f"top_k={top_k}",
                f"experts={num_experts}",
                f"assignments={assignments}",
                f"dtype={str(dtype).replace('torch.', '')}",
                f"reference_us={reference_latency * 1e6:.2f}",
                f"triton_us={triton_latency * 1e6:.2f}",
                f"speedup={reference_latency / triton_latency:.3f}",
                f"triton_assignments_per_s={assignments / triton_latency:.2f}",
                f"peak_memory_bytes={peak_memory}",
                f"triton_kernel_events={kernel_count}",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DWDP grouped expert GEMM.")
    parser.add_argument("--tokens", type=parse_ints, default=(512, 4096, 16384))
    parser.add_argument("--experts", type=parse_ints, default=(8, 16, 64, 128))
    parser.add_argument("--top-k", type=parse_ints, default=(2, 4, 8))
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.dtype == "bf16" and torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("BF16 benchmark requires Ampere or newer")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    for tokens in args.tokens:
        for num_experts in args.experts:
            for top_k in args.top_k:
                benchmark_case(
                    tokens,
                    top_k,
                    num_experts,
                    args.hidden_size,
                    args.output_size,
                    dtype,
                    args.warmup,
                    args.iterations,
                )


if __name__ == "__main__":
    main()
