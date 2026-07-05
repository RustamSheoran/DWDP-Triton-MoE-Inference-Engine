from __future__ import annotations

import argparse
import time

import torch

from DWDP.runtime import DWDPRuntime, RuntimeConfig


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_generate(runtime, tokenizer, prompt: str, max_new_tokens: int, warmup: int, iters: int) -> float:
    inputs = tokenizer(prompt, return_tensors="pt") if tokenizer is not None else prompt
    for _ in range(warmup):
        if tokenizer is not None:
            runtime.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            runtime.generate(inputs, max_new_tokens=max_new_tokens)
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        if tokenizer is not None:
            runtime.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            runtime.generate(inputs, max_new_tokens=max_new_tokens)
    synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Hugging Face and DWDP adapter generation latency.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dwdp_runtime = DWDPRuntime.from_pretrained(
        args.model,
        config=RuntimeConfig(adapter="huggingface", device=args.device),
        device_map=args.device if args.device != "cpu" else None,
    )
    tokenizer = getattr(dwdp_runtime.adapter, "tokenizer", None)
    latency = time_generate(dwdp_runtime, tokenizer, args.prompt, args.max_new_tokens, args.warmup, args.iters)
    print("backend=dwdp")
    print(f"latency_us={latency * 1e6:.2f}")
    print(f"tokens_per_second={args.max_new_tokens / latency:.2f}")


if __name__ == "__main__":
    main()
