#!/usr/bin/env python3
"""Benchmark native Transformers and the DWDP Hugging Face adapter.

This script is intentionally self-contained so it can be launched by
``scripts/benchmark_colab.sh`` in a fresh Google Colab runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
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


def load_kwargs(mode: str) -> dict[str, Any]:
    return {
        "quantization_config": quantization_config(mode),
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }


def input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


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
        torch.cuda.synchronize()


def benchmark(model: Any, tokenizer: Any, args: argparse.Namespace) -> tuple[dict[str, Any], str, list[int]]:
    inputs = make_inputs(tokenizer, model, args.prompt)
    generation_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
        "latency_ms": latency * 1e3,
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
    print(f"prompt={args.prompt!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading native Hugging Face model...")
    hf_load_start = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs(args.quantization))
    hf_load_time_ms = (time.perf_counter() - hf_load_start) * 1e3
    hf_model.eval()
    hf_metrics, hf_text, hf_token_ids = benchmark(hf_model, tokenizer, args)
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
        **load_kwargs(args.quantization),
    )
    dwdp_load_time_ms = (time.perf_counter() - dwdp_load_start) * 1e3
    dwdp_runtime.eval()
    dwdp_metrics, dwdp_text, dwdp_token_ids = benchmark(dwdp_runtime, tokenizer, args)
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
                tokens_per_second=hf_metrics["tokens_per_second"],
                total_runtime_ms=hf_metrics["latency_ms"],
                memory=MemoryMetrics(peak_gpu_memory_bytes=hf_metrics["peak_gpu_memory_bytes"]),
            ),
            dwdp=BackendPerformance(
                backend="dwdp",
                tokens_per_second=dwdp_metrics["tokens_per_second"],
                total_runtime_ms=dwdp_metrics["latency_ms"],
                memory=MemoryMetrics(peak_gpu_memory_bytes=dwdp_metrics["peak_gpu_memory_bytes"]),
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
        },
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
