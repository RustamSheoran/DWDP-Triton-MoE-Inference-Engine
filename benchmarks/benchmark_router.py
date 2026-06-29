from __future__ import annotations

import argparse
import time

import torch

from DWDP.router import LinearTopKRouter, RouterConfig


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[name]


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the DWDP MoE router.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = parse_dtype(args.dtype)
    config = RouterConfig(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    router = LinearTopKRouter(config).to(device=args.device, dtype=dtype).eval()

    if args.compile:
        router = torch.compile(router)

    hidden_states = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden_size,
        device=args.device,
        dtype=dtype,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            router(hidden_states)
        synchronize(args.device)

        start = time.perf_counter()
        for _ in range(args.iters):
            router(hidden_states)
        synchronize(args.device)
        elapsed = time.perf_counter() - start

    avg_seconds = elapsed / args.iters
    avg_microseconds = avg_seconds * 1e6
    tokens = args.batch * args.seq_len
    tokens_per_second = tokens / avg_seconds

    print(f"device={args.device} dtype={args.dtype} compile={args.compile}")
    print(f"shape=[{args.batch}, {args.seq_len}, {args.hidden_size}]")
    print(f"experts={args.num_experts} top_k={args.top_k}")
    print(f"avg_latency_us={avg_microseconds:.2f}")
    print(f"tokens_per_second={tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
