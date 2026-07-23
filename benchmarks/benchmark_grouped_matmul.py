#!/usr/bin/env python3
"""Reproducible kernel-only benchmark for DWDP grouped expert GEMM.

This benchmark intentionally excludes the Router, Dispatcher, Scheduler,
Comms Planner, Executor orchestration, Merger, attention, tokenization, and
generation. It compares only the reference sequential PyTorch expert GEMM
against the Triton grouped expert GEMM kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from DWDP.benchmarking.environment import collect_environment_metadata
from DWDP.executor.kernels import grouped_matmul, reference_grouped_matmul


@dataclass(frozen=True, slots=True)
class KernelCase:
    """One grouped-GEMM experiment configuration."""

    name: str
    num_experts: int
    tokens: int
    top_k: int
    hidden_size: int
    output_size: int
    distribution: str
    dtype: str
    seed: int

    @property
    def assignments(self) -> int:
        """Total routed token-expert assignments."""

        return self.tokens * self.top_k


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Measured kernel-only metrics for one backend."""

    backend: str
    median_latency_ms: float
    mean_latency_ms: float
    assignments_per_second: float
    tokens_per_second: float
    theoretical_flops: int
    achieved_tflops: float
    utilization_estimate_percent: float | None
    cuda_kernel_events: int | None
    allocated_gpu_bytes: int
    peak_gpu_bytes: int
    peak_incremental_gpu_bytes: int


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Numerical comparison of the two grouped GEMM implementations."""

    torch_allclose: bool
    max_absolute_error: float
    mean_absolute_error: float
    rtol: float
    atol: float


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_distribution_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"balanced", "skewed"}
    if not values or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError("distributions must be balanced, skewed, or both")
    return values


def parse_args() -> argparse.Namespace:
    """Parse standalone grouped-GEMM benchmark arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("custom", "qwen", "mixtral"), default="custom")
    parser.add_argument("--experts", type=_parse_int_list, default=(64,))
    parser.add_argument("--tokens", type=_parse_int_list, default=(4096,))
    parser.add_argument("--top-k", type=_parse_int_list, default=(4,))
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--output-size", "--intermediate-size", dest="output_size", type=int, default=5632)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--distributions", type=_parse_distribution_list, default=("balanced", "skewed"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--results-root", default="results/grouped_matmul")
    args = parser.parse_args()
    if args.hidden_size <= 0 or args.output_size <= 0:
        parser.error("--hidden-size and --output-size must be > 0")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")
    return args


def _preset_dimensions(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int, str]:
    if args.preset == "qwen":
        return (64,), args.tokens, (4,), 2048, 5632, "qwen_style"
    if args.preset == "mixtral":
        return (8,), args.tokens, (2,), 4096, 14336, "mixtral_style"
    return args.experts, args.tokens, args.top_k, args.hidden_size, args.output_size, "custom"


def build_cases(args: argparse.Namespace) -> tuple[KernelCase, ...]:
    """Construct reproducible balanced and skewed benchmark cases."""

    experts, tokens, top_k, hidden_size, output_size, prefix = _preset_dimensions(args)
    cases: list[KernelCase] = []
    case_index = 0
    for num_experts in experts:
        for num_tokens in tokens:
            for routing_top_k in top_k:
                for distribution in args.distributions:
                    cases.append(
                        KernelCase(
                            name=f"{prefix}_e{num_experts}_t{num_tokens}_k{routing_top_k}_{distribution}",
                            num_experts=num_experts,
                            tokens=num_tokens,
                            top_k=routing_top_k,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            distribution=distribution,
                            dtype=args.dtype,
                            seed=args.seed + case_index,
                        )
                    )
                    case_index += 1
    return tuple(cases)


def make_expert_counts(num_assignments: int, num_experts: int, distribution: str, device: torch.device) -> torch.Tensor:
    """Build a deterministic balanced or strongly skewed routed-token histogram."""

    if distribution == "balanced":
        counts = torch.full((num_experts,), num_assignments // num_experts, dtype=torch.int64, device=device)
        counts[: num_assignments % num_experts] += 1
        return counts
    if distribution != "skewed":
        raise ValueError(f"unsupported routing distribution '{distribution}'")

    # Mirrors a practical hot-expert tail: 512, 256, 128, 64, 32, 8, then
    # empty experts. The ratios are rescaled to the requested assignment count.
    base = torch.tensor((512, 256, 128, 64, 32, 8), dtype=torch.float64, device=device)
    active = min(num_experts, base.numel())
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    scaled = torch.floor(base[:active] / base[:active].sum() * num_assignments).to(torch.int64)
    counts[:active] = scaled
    counts[0] += num_assignments - int(scaled.sum().item())
    return counts


def make_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Create dispatcher-compatible exclusive expert offsets."""

    return torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)


def _peak_tensor_tflops(gpu_name: str, dtype: torch.dtype) -> float | None:
    """Return approximate dense Tensor Core peak for common target GPUs."""

    name = gpu_name.upper()
    if "T4" in name:
        return 65.13 if dtype == torch.float16 else None
    if "L4" in name:
        return 121.0
    if "A100" in name:
        return 312.0
    if "H100" in name or "H200" in name:
        return 989.0
    return None


def _cuda_event_latencies(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> list[float]:
    """Measure repeated calls with CUDA events, excluding setup and compilation."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def _cuda_kernel_events(fn: Callable[[], torch.Tensor]) -> int | None:
    """Count CUDA profiler events in one post-warmup execution."""

    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        fn()
    total = 0
    for event in profiler.key_averages():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" in device_type:
            total += int(event.count)
    return total


def _measure_backend(
    *,
    backend: str,
    fn: Callable[[], torch.Tensor],
    case: KernelCase,
    warmup: int,
    iterations: int,
    gpu_peak_tflops: float | None,
) -> TimingResult:
    """Measure steady-state kernel-only latency and memory behavior."""

    # Warmup is deliberately outside memory/timing accounting: it compiles and
    # autotunes Triton, initializes cuBLAS, and exercises all preallocated IO.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    latencies = _cuda_event_latencies(fn, warmup=0, iterations=iterations)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    median_latency = statistics.median(latencies)
    mean_latency = statistics.fmean(latencies)
    seconds = median_latency / 1e3
    theoretical_flops = 2 * case.assignments * case.hidden_size * case.output_size
    achieved_tflops = theoretical_flops / seconds / 1e12
    utilization = None if gpu_peak_tflops is None else achieved_tflops / gpu_peak_tflops * 100.0
    return TimingResult(
        backend=backend,
        median_latency_ms=median_latency,
        mean_latency_ms=mean_latency,
        assignments_per_second=case.assignments / seconds,
        tokens_per_second=case.tokens / seconds,
        theoretical_flops=theoretical_flops,
        achieved_tflops=achieved_tflops,
        utilization_estimate_percent=utilization,
        cuda_kernel_events=_cuda_kernel_events(fn),
        allocated_gpu_bytes=allocated_before,
        peak_gpu_bytes=peak_bytes,
        peak_incremental_gpu_bytes=max(peak_bytes - allocated_before, 0),
    )


def _validate_correctness(reference: torch.Tensor, candidate: torch.Tensor, dtype: torch.dtype) -> CorrectnessResult:
    rtol, atol = _tolerance(dtype)
    difference = (reference.float() - candidate.float()).abs()
    result = CorrectnessResult(
        torch_allclose=bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol)),
        max_absolute_error=float(difference.max().item()) if difference.numel() else 0.0,
        mean_absolute_error=float(difference.mean().item()) if difference.numel() else 0.0,
        rtol=rtol,
        atol=atol,
    )
    if not result.torch_allclose:
        raise RuntimeError(
            "Grouped GEMM parity failed: "
            f"max_abs={result.max_absolute_error:.6g}, mean_abs={result.mean_absolute_error:.6g}"
        )
    return result


def benchmark_case(
    case: KernelCase,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Run correctness validation and timed comparison for one real kernel case."""

    dtype = _dtype(case.dtype)
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability(device) < (8, 0):
        raise RuntimeError("BF16 benchmark requires Ampere or newer")
    torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)
    counts = make_expert_counts(case.assignments, case.num_experts, case.distribution, device)
    offsets = make_offsets(counts)
    max_tokens_per_expert = int(counts.max().item())

    # All tensors are allocated and packed before correctness, warmup, and timing.
    inputs = torch.randn(case.assignments, case.hidden_size, device=device, dtype=dtype)
    weights = torch.randn(case.num_experts, case.output_size, case.hidden_size, device=device, dtype=dtype)
    reference_out = torch.empty(case.assignments, case.output_size, device=device, dtype=dtype)
    triton_out = torch.empty_like(reference_out)

    def reference_fn() -> torch.Tensor:
        return reference_grouped_matmul(inputs, weights, offsets, out=reference_out)

    def triton_fn() -> torch.Tensor:
        return grouped_matmul(
            inputs,
            weights,
            offsets,
            max_tokens_per_expert=max_tokens_per_expert,
            out=triton_out,
        )

    correctness = _validate_correctness(reference_fn(), triton_fn(), dtype)
    gpu_name = torch.cuda.get_device_name(device)
    peak_tflops = _peak_tensor_tflops(gpu_name, dtype)
    reference = _measure_backend(
        backend="pytorch_sequential_expert_gemm",
        fn=reference_fn,
        case=case,
        warmup=warmup,
        iterations=iterations,
        gpu_peak_tflops=peak_tflops,
    )
    triton = _measure_backend(
        backend="triton_grouped_expert_gemm",
        fn=triton_fn,
        case=case,
        warmup=warmup,
        iterations=iterations,
        gpu_peak_tflops=peak_tflops,
    )
    return {
        "case": asdict(case),
        "expert_counts": [int(value) for value in counts.cpu().tolist()],
        "correctness": asdict(correctness),
        "pytorch": asdict(reference),
        "triton": asdict(triton),
        "speedup": reference.median_latency_ms / triton.median_latency_ms,
    }


def _create_result_directory(results_root: str | Path) -> Path:
    root = Path(results_root)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = root / stamp
    index = 1
    destination = candidate
    while destination.exists():
        destination = root / f"{stamp}_{index:03d}"
        index += 1
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(config: dict[str, object], environment: dict[str, object], results: list[dict[str, object]]) -> str:
    """Render a self-contained human-readable kernel benchmark report."""

    lines = ["# Grouped Expert GEMM Benchmark", "", "## Environment", "", "| Field | Value |", "| --- | --- |"]
    for key in ("gpu_model", "gpu_memory_bytes", "cuda_version", "cudnn_version", "pytorch_version", "triton_version", "nvidia_driver_version", "git_commit_hash"):
        lines.append(f"| {key} | {_format(environment.get(key))} |")
    lines.extend(["", "## Kernel Configuration", "", "```json", json.dumps(config, indent=2, sort_keys=True), "```", "", "## Results", ""])
    lines.append("| Case | Distribution | Active/Experts | Max Count | Backend | Median ms | Speedup | FLOPs | Assignments/s | Tokens/s | TFLOPS | Utilization % | CUDA events | Peak bytes |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        case = result["case"]
        counts = result["expert_counts"]
        active_experts = sum(count > 0 for count in counts)
        max_count = max(counts, default=0)
        for key, speedup in (("pytorch", 1.0), ("triton", result["speedup"])):
            metrics = result[key]
            lines.append(
                "| {case} | {distribution} | {active}/{experts} | {max_count} | {backend} | {latency} | {speedup} | {flops} | {assignments} | {tokens} | {tflops} | {utilization} | {events} | {memory} |".format(
                    case=case["name"],
                    distribution=case["distribution"],
                    active=active_experts,
                    experts=case["num_experts"],
                    max_count=max_count,
                    backend=metrics["backend"],
                    latency=_format(metrics["median_latency_ms"]),
                    speedup=_format(speedup),
                    flops=_format(metrics["theoretical_flops"], 0),
                    assignments=_format(metrics["assignments_per_second"], 1),
                    tokens=_format(metrics["tokens_per_second"], 1),
                    tflops=_format(metrics["achieved_tflops"]),
                    utilization=_format(metrics["utilization_estimate_percent"]),
                    events=_format(metrics["cuda_kernel_events"]),
                    memory=_format(metrics["peak_gpu_bytes"], 0),
                )
            )
    lines.extend(["", "## Correctness", "", "| Case | torch.allclose | Max absolute error | Mean absolute error |", "| --- | --- | ---: | ---: |"])
    for result in results:
        correctness = result["correctness"]
        lines.append(
            f"| {result['case']['name']} | {correctness['torch_allclose']} | "
            f"{_format(correctness['max_absolute_error'], 6)} | {_format(correctness['mean_absolute_error'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Timings use CUDA events and median steady-state latency.",
            "- Inputs, physical packed weights, offsets, and output buffers are allocated before timing.",
            "- Triton compilation/autotuning and cuBLAS initialization complete during warmup.",
            "- Weight packing/materialization is excluded from timing.",
            "- The utilization estimate is achieved TFLOPS divided by an approximate dense Tensor Core peak when known.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the grouped GEMM benchmark")
    device = torch.device("cuda")
    cases = build_cases(args)
    environment = asdict(
        collect_environment_metadata(
            runtime_backend="kernel_grouped_matmul",
            precision=args.dtype,
            torch_compile=False,
        )
    )
    config = {
        "benchmark": "grouped_expert_gemm",
        "reference_backend": "pytorch_sequential_expert_gemm",
        "optimized_backend": "triton_grouped_expert_gemm",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "preset": args.preset,
        "cases": [asdict(case) for case in cases],
        "timing_excludes": ["allocation", "weight_materialization", "triton_compilation", "triton_autotuning", "python_setup"],
    }
    results: list[dict[str, object]] = []
    for case in cases:
        print(f"benchmarking={case.name}")
        results.append(benchmark_case(case, device, warmup=args.warmup, iterations=args.iterations))

    output_dir = _create_result_directory(args.results_root)
    report_payload = {"config": config, "environment": environment, "results": results}
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "report.json", report_payload)
    (output_dir / "report.md").write_text(render_report(config, environment, results), encoding="utf-8")
    print(f"results_dir={output_dir}")


if __name__ == "__main__":
    main()
