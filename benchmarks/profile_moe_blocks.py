#!/usr/bin/env python3
"""Profile one native Hugging Face MoE block against its DWDP replacement.

This is profiling infrastructure, not an optimization benchmark. It compares
the same Qwen-style MoE weights and identical hidden states without patching
the loaded Hugging Face model. DWDP stages are timed with CUDA events; native
HF is reported as a monolithic block plus module-hook diagnostics because it
does not expose DWDP's dispatcher, scheduler, communication planner, or
merger boundaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM

from DWDP.adapters.extractor import discover_qwen_moe_layers
from DWDP.adapters.qwen15_moe import DWDPMoEBlock
from DWDP.benchmarking.environment import collect_environment_metadata
from DWDP.runtime import RuntimeConfig


DWDP_STAGE_NAMES = ("router", "dispatcher", "scheduler", "comms_planner", "executor", "merger")
SYNC_PATTERNS = ("aten::item", "aten::_local_scalar_dense", "synchronize", "cuda_synchronize")


@dataclass(slots=True)
class StageEvent:
    """One asynchronous DWDP stage measurement."""

    name: str
    cpu_orchestration_ms: float
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Aggregated CUDA-event and CPU orchestration timing for one stage."""

    name: str
    gpu_latency_ms: float
    cpu_orchestration_ms: float


@dataclass(frozen=True, slots=True)
class ProfileDetails:
    """Torch-profiler metadata for one callable or DWDP stage."""

    cuda_kernel_events: int
    synchronization_events: int
    allocated_before_bytes: int
    allocated_after_bytes: int
    allocation_delta_bytes: int
    peak_gpu_bytes: int
    top_operators: tuple[dict[str, object], ...]


def parse_args() -> argparse.Namespace:
    """Parse reproducible MoE block profiling options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results-root", default="results/profile")
    args = parser.parse_args()
    if args.layer_index < 0 or args.batch_size <= 0 or args.sequence_length <= 0:
        parser.error("--layer-index must be >= 0; --batch-size and --sequence-length must be > 0")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")
    return args


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _unwrap_hidden_states(output: object) -> torch.Tensor:
    """Extract hidden states from common Hugging Face MoE block outputs."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported MoE block output type: {type(output)!r}")


def _cuda_event_total(fn: Callable[[], object]) -> tuple[float, float, object]:
    """Measure one callable without synchronizing before its GPU work queues."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.perf_counter()
    start.record()
    output = fn()
    end.record()
    cpu_ms = (time.perf_counter() - cpu_start) * 1e3
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), cpu_ms, output


def _run_dwdp_staged(
    block: DWDPMoEBlock,
    hidden_states: torch.Tensor,
    *,
    stage_wrapper: Callable[[str, Callable[[], object]], object] | None = None,
) -> tuple[torch.Tensor, tuple[StageEvent, ...]]:
    """Run the unchanged DWDP MoE pipeline with CUDA-event stage boundaries."""

    workspaces = block.context.workspaces
    records: list[StageEvent] = []

    def stage(name: str, fn: Callable[[], object]) -> object:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.perf_counter()
        start.record()
        with torch.autograd.profiler.record_function(f"profile.dwdp.{name}"):
            value = stage_wrapper(name, fn) if stage_wrapper is not None else fn()
        end.record()
        records.append(StageEvent(name, (time.perf_counter() - cpu_start) * 1e3, start, end))
        return value

    router_output = stage("router", lambda: block.router(hidden_states))
    dispatch_plan = stage(
        "dispatcher",
        lambda: block.dispatcher(router_output, workspace=workspaces.dispatch if workspaces is not None else None),
    )
    execution_plan = stage(
        "scheduler",
        lambda: block.scheduler(dispatch_plan, workspace=workspaces.scheduler if workspaces is not None else None),
    )
    communication_plan = stage(
        "comms_planner",
        lambda: block.comms_planner(execution_plan, workspace=workspaces.comms if workspaces is not None else None),
    )
    executor_output = stage(
        "executor",
        lambda: block.executor(
            hidden_states,
            dispatch_plan,
            execution_plan,
            communication_plan,
            workspace=workspaces.executor if workspaces is not None else None,
        ),
    )
    merger_output = stage(
        "merger",
        lambda: block.merger(executor_output, workspace=workspaces.merger if workspaces is not None else None),
    )
    output = merger_output.hidden_states
    if block.shared_expert is not None:
        # Preserve the adapter's exact post-MoE shared-expert behavior. The
        # stage event is extended here because it owns the final output
        # reconstruction and must account for every operation in the block.
        shared_end = torch.cuda.Event(enable_timing=True)
        shared_cpu_start = time.perf_counter()
        shared = block.shared_expert(hidden_states)
        if block.shared_expert_gate is not None:
            shared = torch.sigmoid(block.shared_expert_gate(hidden_states)) * shared
        output = output + shared
        shared_end.record()
        records[-1] = StageEvent(
            "merger",
            records[-1].cpu_orchestration_ms + (time.perf_counter() - shared_cpu_start) * 1e3,
            records[-1].start,
            shared_end,
        )
    return output, tuple(records)


def _aggregate_stage_events(events: tuple[tuple[StageEvent, ...], ...]) -> tuple[StageTiming, ...]:
    """Synchronize once and aggregate event timings across timed iterations."""

    torch.cuda.synchronize()
    result: list[StageTiming] = []
    for name in DWDP_STAGE_NAMES:
        matching = [record for iteration in events for record in iteration if record.name == name]
        result.append(
            StageTiming(
                name=name,
                gpu_latency_ms=statistics.median(record.start.elapsed_time(record.end) for record in matching),
                cpu_orchestration_ms=statistics.median(record.cpu_orchestration_ms for record in matching),
            )
        )
    return tuple(result)


def _run_iterations(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    staged: bool,
) -> tuple[float, float, tuple[tuple[StageEvent, ...], ...]]:
    """Return median full-block timing and optional DWDP stage event records."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    totals: list[float] = []
    cpu_totals: list[float] = []
    stage_records: list[tuple[StageEvent, ...]] = []
    for _ in range(iterations):
        if staged:
            gpu_ms, cpu_ms, value = _cuda_event_total(fn)
            _, records = value
            stage_records.append(records)
        else:
            gpu_ms, cpu_ms, _ = _cuda_event_total(fn)
        totals.append(gpu_ms)
        cpu_totals.append(cpu_ms)
    return statistics.median(totals), statistics.median(cpu_totals), tuple(stage_records)


def _profile_callable(name: str, fn: Callable[[], object], *, with_stack: bool) -> tuple[object, ProfileDetails]:
    """Capture kernel count, synchronization indicators, allocations, and operators."""

    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=with_stack,
        record_shapes=True,
    ) as profiler:
        with torch.autograd.profiler.record_function(name):
            value = fn()
    torch.cuda.synchronize()
    events = profiler.key_averages()
    trace_events = profiler.events()
    kernel_events = sum(
        1 for event in trace_events if "cuda" in str(getattr(event, "device_type", "")).lower()
    )
    synchronization_events = 0
    for event in events:
        key = str(getattr(event, "key", "")).lower()
        if any(pattern in key for pattern in SYNC_PATTERNS):
            synchronization_events += int(event.count)
    top_operators = tuple(
        {
            "operator": event.key,
            "calls": int(event.count),
            "self_cpu_ms": float(getattr(event, "self_cpu_time_total", 0.0)) / 1e3,
            "self_cuda_ms": float(
                getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
            )
            / 1e3,
            "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
            "self_cuda_memory_bytes": int(
                getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))
            ),
        }
        for event in sorted(events, key=lambda item: float(getattr(item, "self_cpu_time_total", 0.0)), reverse=True)[:25]
    )
    allocated_after = int(torch.cuda.memory_allocated())
    return value, ProfileDetails(
        cuda_kernel_events=kernel_events,
        synchronization_events=synchronization_events,
        allocated_before_bytes=allocated_before,
        allocated_after_bytes=allocated_after,
        allocation_delta_bytes=allocated_after - allocated_before,
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
        top_operators=top_operators,
    )


def _export_combined_trace(
    path: Path,
    native_fn: Callable[[], object],
    dwdp_fn: Callable[[], object],
    *,
    with_stack: bool,
) -> None:
    """Export one Chrome trace containing native and named DWDP pipeline ranges."""

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=with_stack,
        record_shapes=True,
    ) as profiler:
        with torch.autograd.profiler.record_function("profile.native_hf_moe_block"):
            native_fn()
        with torch.autograd.profiler.record_function("profile.dwdp_moe_block"):
            dwdp_fn()
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))


def _native_hook_diagnostics(native_block: torch.nn.Module, hidden_states: torch.Tensor) -> dict[str, float]:
    """Measure native gate and expert module GPU work without rewriting HF logic."""

    records: dict[str, list[tuple[float, torch.cuda.Event, torch.cuda.Event]]] = {"hf_gate_projection": [], "hf_expert_modules": []}

    def make_hooks(category: str):
        starts: list[tuple[float, torch.cuda.Event]] = []

        def pre_hook(_module: torch.nn.Module, _inputs: tuple[object, ...]) -> None:
            event = torch.cuda.Event(enable_timing=True)
            cpu_start = time.perf_counter()
            event.record()
            starts.append((cpu_start, event))

        def post_hook(_module: torch.nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            cpu_start, start = starts.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            records[category].append(((time.perf_counter() - cpu_start) * 1e3, start, end))

        return pre_hook, post_hook

    handles = []
    gate = getattr(native_block, "gate", None)
    if isinstance(gate, torch.nn.Module):
        pre_hook, post_hook = make_hooks("hf_gate_projection")
        handles.extend((gate.register_forward_pre_hook(pre_hook), gate.register_forward_hook(post_hook)))
    for expert in getattr(native_block, "experts", ()):
        if isinstance(expert, torch.nn.Module):
            pre_hook, post_hook = make_hooks("hf_expert_modules")
            handles.extend((expert.register_forward_pre_hook(pre_hook), expert.register_forward_hook(post_hook)))
    try:
        native_block(hidden_states)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    result: dict[str, float] = {}
    for category, values in records.items():
        result[f"{category}_gpu_ms"] = sum(start.elapsed_time(end) for _, start, end in values)
        result[f"{category}_cpu_ms"] = sum(cpu_ms for cpu_ms, _, _ in values)
    return result


def _create_result_directory(results_root: str | Path) -> Path:
    root = Path(results_root)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result = root / stamp
    suffix = 1
    while result.exists():
        result = root / f"{stamp}_{suffix:03d}"
        suffix += 1
    result.mkdir(parents=True, exist_ok=False)
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_report(payload: dict[str, object]) -> str:
    environment = payload["environment"]
    config = payload["config"]
    native = payload["native_hf"]
    dwdp = payload["dwdp"]
    correctness = payload["correctness"]
    lines = ["# HF vs DWDP MoE Block Profile", "", "## Environment", "", "| Field | Value |", "| --- | --- |"]
    for key in ("gpu_model", "gpu_memory_bytes", "cuda_version", "pytorch_version", "triton_version", "nvidia_driver_version", "git_commit_hash"):
        lines.append(f"| {key} | {_fmt(environment.get(key))} |")
    lines.extend(["", "## Configuration", "", "```json", json.dumps(config, indent=2, sort_keys=True), "```", "", "## End-to-End MoE Block", ""])
    lines.append("| Backend | Median GPU ms | Median CPU orchestration ms | CUDA kernel events | Sync events | Peak GPU bytes |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, item in (("Native Hugging Face", native), ("DWDP patched block", dwdp)):
        profile = item["profile"]
        lines.append(
            f"| {label} | {_fmt(item['gpu_latency_ms'])} | {_fmt(item['cpu_orchestration_ms'])} | "
            f"{_fmt(profile['cuda_kernel_events'])} | {_fmt(profile['synchronization_events'])} | {_fmt(profile['peak_gpu_bytes'], 0)} |"
        )
    lines.extend(["", "## DWDP Stage Timeline", "", "| Stage | GPU ms | CPU orchestration ms | CUDA kernels | Sync events | Allocation delta bytes | Peak GPU bytes |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for timing in dwdp["stages"]:
        details = dwdp["stage_profiles"][timing["name"]]
        lines.append(
            f"| {timing['name']} | {_fmt(timing['gpu_latency_ms'])} | {_fmt(timing['cpu_orchestration_ms'])} | "
            f"{_fmt(details['cuda_kernel_events'])} | {_fmt(details['synchronization_events'])} | "
            f"{_fmt(details['allocation_delta_bytes'], 0)} | {_fmt(details['peak_gpu_bytes'], 0)} |"
        )
    lines.extend(["", "## Native HF Diagnostics", "", "Native HF does not expose DWDP dispatcher, scheduler, communication planner, or merger stages. The following are module-hook diagnostics, not one-to-one stage equivalents.", ""])
    lines.append("| Native component | GPU ms | CPU hook ms |")
    lines.append("| --- | ---: | ---: |")
    for name, value in native["hook_diagnostics"].items():
        if name.endswith("_gpu_ms"):
            cpu_name = name.replace("_gpu_ms", "_cpu_ms")
            lines.append(f"| {name.removesuffix('_gpu_ms')} | {_fmt(value)} | {_fmt(native['hook_diagnostics'].get(cpu_name))} |")
    lines.extend(["", "## Correctness", "", "| Metric | Value |", "| --- | --- |"])
    for key in ("torch_allclose", "max_absolute_error", "mean_absolute_error", "rtol", "atol", "shared_parameter_storage"):
        lines.append(f"| {key} | {_fmt(correctness.get(key), 6)} |")
    lines.extend(["", "## Where DWDP Loses", ""])
    total_gap = dwdp["gpu_latency_ms"] - native["gpu_latency_ms"]
    lines.append(f"- Native HF median block GPU latency: `{native['gpu_latency_ms']:.3f} ms`.")
    lines.append(f"- DWDP median block GPU latency: `{dwdp['gpu_latency_ms']:.3f} ms`.")
    lines.append(f"- DWDP minus native GPU latency: `{total_gap:.3f} ms`.")
    lines.append("- Attribute the gap using the DWDP stage table and the top operators in `report.json` / `profiler_trace.json`; do not treat unavailable HF stage rows as zero-cost work.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for MoE block profiling")
    dtype = _dtype(args.dtype)
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        raise SystemExit("BF16 profiling requires Ampere or newer")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).cuda().eval()
    specs = discover_qwen_moe_layers(model)
    if not specs:
        raise SystemExit("No supported Qwen-style MoE blocks were discovered")
    if args.layer_index >= len(specs):
        raise SystemExit(f"--layer-index {args.layer_index} is outside discovered range [0, {len(specs) - 1}]")
    spec = specs[args.layer_index]
    native_block = spec.module
    dwdp_block = DWDPMoEBlock(
        spec,
        RuntimeConfig(
            backend="dwdp_profile",
            device="cuda",
            dtype=dtype,
            deterministic=True,
            enable_workspace=True,
            enable_profiling=False,
        ),
    ).eval()
    hidden_states = torch.randn(args.batch_size, args.sequence_length, spec.hidden_size, device="cuda", dtype=dtype)

    @torch.inference_mode()
    def native_fn() -> torch.Tensor:
        return _unwrap_hidden_states(native_block(hidden_states))

    @torch.inference_mode()
    def dwdp_fn() -> tuple[torch.Tensor, tuple[StageEvent, ...]]:
        return _run_dwdp_staged(dwdp_block, hidden_states)

    # Same module objects are used for native and DWDP expert execution.
    shared_parameter_storage = all(
        dwdp_block.executor.experts.get(expert_id) is spec.experts[expert_id]
        for expert_id in dwdp_block.executor.experts.expert_ids
    )
    native_output = native_fn()
    dwdp_output, _ = dwdp_fn()
    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)
    difference = (native_output.float() - dwdp_output.float()).abs()
    correctness = {
        "torch_allclose": bool(torch.allclose(native_output, dwdp_output, rtol=rtol, atol=atol)),
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "rtol": rtol,
        "atol": atol,
        "shared_parameter_storage": shared_parameter_storage,
    }
    if not correctness["torch_allclose"]:
        raise RuntimeError(f"Native HF and DWDP MoE block parity failed: {correctness}")

    native_gpu_ms, native_cpu_ms, _ = _run_iterations(native_fn, warmup=args.warmup, iterations=args.iterations, staged=False)
    dwdp_gpu_ms, dwdp_cpu_ms, stage_events = _run_iterations(dwdp_fn, warmup=args.warmup, iterations=args.iterations, staged=True)
    stage_timings = _aggregate_stage_events(stage_events)

    _, native_profile = _profile_callable("profile.native_hf_moe_block", native_fn, with_stack=args.with_stack)
    stage_profiles: dict[str, ProfileDetails] = {}

    # Each profile captures the real stage invocation in one dependency-valid
    # pipeline pass. CUDA events above remain the timing source of truth.
    def profile_stage(name: str, fn: Callable[[], object]) -> object:
        value, details = _profile_callable(f"profile.dwdp.{name}", fn, with_stack=args.with_stack)
        stage_profiles[name] = details
        return value

    _run_dwdp_staged(dwdp_block, hidden_states, stage_wrapper=profile_stage)
    _, dwdp_profile = _profile_callable("profile.dwdp_moe_block", dwdp_fn, with_stack=args.with_stack)
    hook_diagnostics = _native_hook_diagnostics(native_block, hidden_states)

    output_dir = _create_result_directory(args.results_root)
    _export_combined_trace(output_dir / "profiler_trace.json", native_fn, dwdp_fn, with_stack=args.with_stack)
    environment = asdict(collect_environment_metadata(runtime_backend="dwdp_profile", precision=args.dtype, torch_compile=False))
    config = {
        "model": args.model,
        "layer_index": args.layer_index,
        "layer_name": spec.name,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": spec.hidden_size,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "with_stack": args.with_stack,
        "identical_model_weights": True,
        "identical_hidden_states": True,
    }
    payload = {
        "config": config,
        "environment": environment,
        "correctness": correctness,
        "native_hf": {
            "gpu_latency_ms": native_gpu_ms,
            "cpu_orchestration_ms": native_cpu_ms,
            "profile": asdict(native_profile),
            "hook_diagnostics": hook_diagnostics,
        },
        "dwdp": {
            "gpu_latency_ms": dwdp_gpu_ms,
            "cpu_orchestration_ms": dwdp_cpu_ms,
            "profile": asdict(dwdp_profile),
            "stages": [asdict(item) for item in stage_timings],
            "stage_profiles": {name: asdict(details) for name, details in stage_profiles.items()},
        },
        "trace": "profiler_trace.json",
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "report.json", payload)
    (output_dir / "report.md").write_text(_render_report(payload), encoding="utf-8")
    print(f"results_dir={output_dir}")


if __name__ == "__main__":
    main()
