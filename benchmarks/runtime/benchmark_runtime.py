from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from DWDP.runtime import DWDPRuntime, RuntimeConfig


class MLPExpert(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the integrated DWDP reference runtime.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    experts = [MLPExpert(args.hidden_size).to(args.device) for _ in range(args.num_experts)]
    runtime = DWDPRuntime.build_reference(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        experts=experts,
        config=RuntimeConfig(device=args.device, enable_workspace=True),
    ).to(args.device)
    hidden_states = torch.randn(args.batch, args.seq_len, args.hidden_size, device=args.device)

    with torch.no_grad():
        for _ in range(args.warmup):
            runtime(hidden_states)
        synchronize(args.device)
        start = time.perf_counter()
        for _ in range(args.iters):
            runtime(hidden_states)
        synchronize(args.device)

    latency = (time.perf_counter() - start) / args.iters
    tokens = args.batch * args.seq_len
    print(f"device={args.device}")
    print(f"tokens={tokens} hidden_size={args.hidden_size} experts={args.num_experts} top_k={args.top_k}")
    print(f"latency_us={latency * 1e6:.2f}")
    print(f"tokens_per_second={tokens / latency:.2f}")
    print(f"workspace_bytes={runtime.context.workspaces.estimated_bytes() if runtime.context.workspaces else 0}")


if __name__ == "__main__":
    main()
