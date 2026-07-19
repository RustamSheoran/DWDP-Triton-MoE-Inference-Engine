#!/usr/bin/env python3
"""Benchmark native Transformers and the DWDP Hugging Face adapter.

This script is intentionally self-contained so it can be launched by
``scripts/benchmark_colab.sh`` in a fresh Google Colab runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from DWDP.benchmarking import (
    BackendPerformance,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkReportWriter,
    CorrectnessMetrics,
    GenerationConfig,
    MemoryMetrics,
    PerformanceComparison,
    ReportMetadata,
    RuntimeBreakdown,
    RuntimeStatistics,
)
from DWDP.benchmarking.environment import collect_environment_metadata
from DWDP.runtime import DWDPRuntime, RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="4bit")
    parser.add_argument("--hf-token", "--use", dest="hf_token", default=None, help="Hugging Face token; also read from HF_TOKEN.")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=True,
        help="Collect the detailed Torch profiler pass (default).",
    )
    profile_group.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Skip detailed profiling for a faster benchmark.",
    )
    parser.add_argument("--results-root", default="results", help="Directory for timestamped benchmark reports.")
    parser.add_argument("--output-json", default=None, help="Optional path for machine-readable results.")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("--max-new-tokens and --iters must be > 0; --warmup must be >= 0")
    return args


def quantization_config(mode: str) -> BitsAndBytesConfig:
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_kwargs(mode: str, token: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "quantization_config": quantization_config(mode),
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if token:
        kwargs["token"] = token
    return kwargs


def input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def resolve_hf_token(explicit_token: str | None) -> str | None:
    """Resolve a token from CLI, environment, or Colab Secrets."""

    if explicit_token:
        return explicit_token
    environment_token = os.environ.get("HF_TOKEN")
    if environment_token:
        return environment_token
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN") or None
    except Exception:
        return None


def make_inputs(tokenizer: Any, model: Any, prompt: str) -> dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = input_device(model)
    return {key: value.to(device) for key, value in inputs.items()}


def synchronize() -> None:
    if torch.cuda.is_available():
        with torch.autograd.profiler.record_function("benchmark.synchronization"):
            torch.cuda.synchronize()


def profile_generation(model: Any, tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Collect a one-generation Torch profiler summary by useful categories."""

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    inputs = make_inputs(tokenizer, model, args.prompt)
    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    categories = {
        "python_orchestration": ("dwdp.python_orchestration",),
        "router": ("dwdp.router",),
        "dispatcher": ("dwdp.dispatcher",),
        "scheduler": ("dwdp.scheduler",),
        "comms_planner": ("dwdp.comms_planner",),
        "executor": ("dwdp.executor",),
        "merger": ("dwdp.merger",),
        "gather": ("dwdp.gather", "aten::index", "aten::index_select"),
        "gemms": ("dwdp.expert_gemms", "aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"),
        "copies": ("aten::copy_", "aten::_to_copy", "aten::to"),
        "synchronization": ("benchmark.synchronization", "cuda_synchronize"),
    }
    summary: dict[str, Any] = {}
    events = profiler.key_averages()
    for category, patterns in categories.items():
        matching = [event for event in events if any(pattern in event.key.lower() for pattern in patterns)]
        summary[category] = {
            "cpu_ms": sum(float(getattr(event, "self_cpu_time_total", 0.0)) for event in matching) / 1000.0,
            "device_ms": sum(
                float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)))
                for event in matching
            )
            / 1000.0,
            "operators": sorted({event.key for event in matching}),
        }
    summary["top_operators"] = [
        {
            "operator": event.key,
            "self_cpu_ms": float(getattr(event, "self_cpu_time_total", 0.0)) / 1000.0,
            "self_device_ms": float(
                getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
            )
            / 1000.0,
            "calls": int(event.count),
        }
        for event in sorted(events, key=lambda item: float(getattr(item, "self_cpu_time_total", 0.0)), reverse=True)[:30]
    ]
    return summary


def measure_prefill(model: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> float:
    """Measure prompt-only forward latency, excluding token sampling/decoding."""

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**inputs, use_cache=True, return_dict=True)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(**inputs, use_cache=True, return_dict=True)
        synchronize()
    return (time.perf_counter() - start) * 1e3 / args.iters


def measure_ttft(model: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> float:
    """Measure time to generate the first token (prefill plus first decode)."""

    with torch.inference_mode():
        for _ in range(args.warmup):
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
        synchronize()
    return (time.perf_counter() - start) * 1e3 / args.iters


def profiled_cpu_ms(profile: dict[str, Any], name: str) -> float | None:
    value = profile.get(name, {}).get("cpu_ms") if profile else None
    return float(value) if value is not None else None


def benchmark(model: Any, tokenizer: Any, args: argparse.Namespace) -> tuple[dict[str, Any], str, list[int]]:
    inputs = make_inputs(tokenizer, model, args.prompt)
    generation_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    prefill_latency_ms = measure_prefill(model, inputs, args)
    ttft_ms = measure_ttft(model, inputs, args)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model.generate(**inputs, **generation_kwargs)
        synchronize()
        start = time.perf_counter()
        output_ids = None
        for _ in range(args.iters):
            output_ids = model.generate(**inputs, **generation_kwargs)
        synchronize()

    latency = (time.perf_counter() - start) / args.iters
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    metrics = {
        "ttft_ms": ttft_ms,
        "prefill_latency_ms": prefill_latency_ms,
        "latency_ms": latency * 1e3,
        "decode_latency_ms": max(latency * 1e3 - ttft_ms, 0.0),
        "tokens_per_second": generated.numel() / latency,
        "input_tokens": inputs["input_ids"].shape[-1],
        "generated_tokens": generated.numel(),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
    }
    return metrics, text, generated.detach().cpu().tolist()


def release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA runtime is required. In Colab select Runtime > Change runtime type > T4 GPU.")

    print(f"model={args.model}")
    print(f"quantization={args.quantization}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    token = resolve_hf_token(args.hf_token)
    print(f"prompt={args.prompt!r}")
    print(f"hf_token={'provided' if token else 'not provided'}")

    tokenizer_kwargs = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading native Hugging Face model...")
    hf_load_start = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs(args.quantization, token))
    hf_load_time_ms = (time.perf_counter() - hf_load_start) * 1e3
    hf_model.eval()
    hf_metrics, hf_text, hf_token_ids = benchmark(hf_model, tokenizer, args)
    hf_profile = profile_generation(hf_model, tokenizer, args) if args.profile else {}
    print(f"hf_latency_ms={hf_metrics['latency_ms']:.2f}")
    print(f"hf_tokens_per_second={hf_metrics['tokens_per_second']:.2f}")
    print(f"hf_output={hf_text!r}")
    release(hf_model)
    # release() clears CUDA's allocator, but the caller must also drop its
    # reference or Python will keep the model (and its VRAM allocations) alive.
    hf_model = None

    print("\nLoading DWDP-patched model...")
    dwdp_load_start = time.perf_counter()
    dwdp_runtime = DWDPRuntime.from_pretrained(
        args.model,
        config=RuntimeConfig(backend="dwdp", device="cuda", dtype=torch.float16),
        **load_kwargs(args.quantization, token),
    )
    dwdp_load_time_ms = (time.perf_counter() - dwdp_load_start) * 1e3
    dwdp_runtime.eval()
    dwdp_metrics, dwdp_text, dwdp_token_ids = benchmark(dwdp_runtime, tokenizer, args)
    dwdp_profile = profile_generation(dwdp_runtime, tokenizer, args) if args.profile else {}
    print(f"dwdp_latency_ms={dwdp_metrics['latency_ms']:.2f}")
    print(f"dwdp_tokens_per_second={dwdp_metrics['tokens_per_second']:.2f}")
    print(f"dwdp_output={dwdp_text!r}")

    results = {
        "model": args.model,
        "quantization": args.quantization,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu": torch.cuda.get_device_name(0),
        "hf": {**hf_metrics, "output": hf_text},
        "dwdp": {**dwdp_metrics, "output": dwdp_text},
    }

    environment = collect_environment_metadata(
        runtime_backend="dwdp_reference",
        precision=args.quantization,
        torch_compile=False,
    )
    report_config = BenchmarkConfig(
        model_name=args.model,
        checkpoint=args.model,
        prompt=args.prompt,
        batch_size=1,
        sequence_length=int(hf_metrics["input_tokens"]),
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False),
        dtype="float16_compute",
        device="cuda",
        random_seed=0,
        backend="hf",
        compare_backend="dwdp",
        runtime_backend="dwdp_reference",
        runtime_config={"quantization": args.quantization, "device_map": "auto"},
    )
    stage_names = ("router", "dispatcher", "scheduler", "comms_planner", "executor", "merger")
    stage_values = {name: profiled_cpu_ms(dwdp_profile, name) for name in stage_names}
    stage_total = sum(value for value in stage_values.values() if value is not None)
    stage_percentages = {
        name: (value / stage_total * 100.0 if value is not None and stage_total else None)
        for name, value in stage_values.items()
    }
    latency_change_pct = (dwdp_metrics["latency_ms"] / hf_metrics["latency_ms"] - 1.0) * 100.0
    throughput_change_pct = (dwdp_metrics["tokens_per_second"] / hf_metrics["tokens_per_second"] - 1.0) * 100.0
    memory_change_pct = (
        (dwdp_metrics["peak_gpu_memory_bytes"] / hf_metrics["peak_gpu_memory_bytes"] - 1.0) * 100.0
        if hf_metrics["peak_gpu_memory_bytes"]
        else None
    )
    speed_word = "faster" if latency_change_pct < 0 else "slower"
    throughput_word = "higher" if throughput_change_pct > 0 else "lower"
    memory_observation = (
        f"DWDP peak GPU memory is {memory_change_pct:+.2f}% versus native Hugging Face."
        if memory_change_pct is not None
        else "Peak GPU memory comparison was unavailable."
    )
    report = BenchmarkReport(
        metadata=ReportMetadata(
            experiment_name="colab_hf_vs_dwdp",
            notes=(
                "Native Transformers and DWDP used the same prompt and generation settings.",
                "DWDP is measured through the current Hugging Face adapter/reference PyTorch path.",
            ),
            tags=("colab", "quantized", args.quantization),
        ),
        config=report_config,
        environment=environment,
        performance=PerformanceComparison(
            huggingface=BackendPerformance(
                backend="hf",
                ttft_ms=hf_metrics["ttft_ms"],
                prefill_latency_ms=hf_metrics["prefill_latency_ms"],
                decode_latency_ms=hf_metrics["decode_latency_ms"],
                tokens_per_second=hf_metrics["tokens_per_second"],
                total_runtime_ms=hf_metrics["latency_ms"],
                memory=MemoryMetrics(peak_gpu_memory_bytes=hf_metrics["peak_gpu_memory_bytes"]),
            ),
            dwdp=BackendPerformance(
                backend="dwdp",
                ttft_ms=dwdp_metrics["ttft_ms"],
                prefill_latency_ms=dwdp_metrics["prefill_latency_ms"],
                decode_latency_ms=dwdp_metrics["decode_latency_ms"],
                tokens_per_second=dwdp_metrics["tokens_per_second"],
                total_runtime_ms=dwdp_metrics["latency_ms"],
                memory=MemoryMetrics(peak_gpu_memory_bytes=dwdp_metrics["peak_gpu_memory_bytes"]),
            ),
            runtime_breakdown=RuntimeBreakdown(
                router_ms=stage_values["router"],
                dispatcher_ms=stage_values["dispatcher"],
                scheduler_ms=stage_values["scheduler"],
                comms_planner_ms=stage_values["comms_planner"],
                executor_ms=stage_values["executor"],
                merger_ms=stage_values["merger"],
                total_dwdp_overhead_ms=stage_total or None,
                module_percentages=stage_percentages,
            ),
        ),
        correctness=CorrectnessMetrics(
            generated_token_parity=hf_token_ids == dwdp_token_ids,
        ),
        runtime_statistics=RuntimeStatistics(
            future_extensions={
                "hf_load_time_ms": hf_load_time_ms,
                "dwdp_load_time_ms": dwdp_load_time_ms,
                "input_tokens": hf_metrics["input_tokens"],
                "generated_tokens": hf_metrics["generated_tokens"],
            }
        ),
        profiler={
            "hf_load_time_ms": hf_load_time_ms,
            "dwdp_load_time_ms": dwdp_load_time_ms,
            "hf_output": hf_text,
            "dwdp_output": dwdp_text,
            "torch_profiler_enabled": args.profile,
            "hf": hf_profile,
            "dwdp": dwdp_profile,
        },
        observations=(
            f"DWDP is {abs(latency_change_pct):.2f}% {speed_word} than native Hugging Face by end-to-end latency.",
            f"DWDP throughput is {abs(throughput_change_pct):.2f}% {throughput_word} than native Hugging Face.",
            memory_observation,
            "Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.",
        ),
    )
    report_paths = BenchmarkReportWriter(results_root=args.results_root).write(report)
    print(f"results_dir={report_paths.root}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"results_json={args.output_json}")
    release(dwdp_runtime)


if __name__ == "__main__":
    main()
