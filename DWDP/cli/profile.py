from __future__ import annotations

import argparse

from DWDP.runtime import DWDPRuntime, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the profile CLI parser."""

    parser = argparse.ArgumentParser(description="Profile a DWDP-compatible generation wrapper.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute the profile CLI."""

    args = build_parser().parse_args(argv)
    runtime = DWDPRuntime.from_pretrained(
        args.model,
        config=RuntimeConfig(device=args.device, enable_profiling=True),
    )
    tokenizer = getattr(runtime.adapter, "tokenizer", None) if hasattr(runtime, "adapter") else None
    if tokenizer is not None:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        runtime.generate(**inputs, max_new_tokens=args.max_new_tokens)
    else:
        runtime.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print("profile_status=delegated_generation")
    print("torch_profiler_placeholder=true")
    print("nsight_placeholder=true")


if __name__ == "__main__":
    main()
