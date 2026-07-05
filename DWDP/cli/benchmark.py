from __future__ import annotations

import argparse
import time

import torch

from DWDP.runtime import DWDPRuntime, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""

    parser = argparse.ArgumentParser(description="Benchmark HF and DWDP runtime wrappers.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hf", choices=("hf", "dwdp"))
    parser.add_argument("--compare", default=None, choices=("hf", "dwdp"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def _time_generation(runtime, tokenizer, prompt: str, max_new_tokens: int, warmup: int, iters: int) -> dict[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt") if tokenizer is not None else prompt
    for _ in range(warmup):
        if tokenizer is not None:
            runtime.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            runtime.generate(inputs, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        if tokenizer is not None:
            runtime.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            runtime.generate(inputs, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / iters
    return {
        "latency_us": latency * 1e6,
        "tokens_per_second": max_new_tokens / latency,
        "first_token_latency_us": latency * 1e6,
    }


def main(argv: list[str] | None = None) -> None:
    """Execute the benchmark CLI."""

    args = build_parser().parse_args(argv)
    primary = DWDPRuntime.from_pretrained(args.model, config=RuntimeConfig(backend=args.backend, device=args.device))
    tokenizer = getattr(primary.adapter, "tokenizer", None) if hasattr(primary, "adapter") else None
    primary_metrics = _time_generation(
        primary,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.warmup,
        args.iters,
    )
    print(f"backend={args.backend}")
    for key, value in primary_metrics.items():
        print(f"{key}={value:.2f}")

    if args.compare is not None:
        compare = DWDPRuntime.from_pretrained(args.model, config=RuntimeConfig(backend=args.compare, device=args.device))
        compare_metrics = _time_generation(
            compare,
            getattr(compare.adapter, "tokenizer", tokenizer) if hasattr(compare, "adapter") else tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.warmup,
            args.iters,
        )
        print(f"compare_backend={args.compare}")
        for key, value in compare_metrics.items():
            print(f"compare_{key}={value:.2f}")


if __name__ == "__main__":
    main()
