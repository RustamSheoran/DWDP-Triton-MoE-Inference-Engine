from __future__ import annotations

import argparse

from DWDP.runtime import DWDPRuntime, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the `dwdp run` parser."""

    parser = argparse.ArgumentParser(description="Run text generation through a DWDP runtime wrapper.")
    parser.add_argument("--model", required=True, help="Hugging Face model path or identifier.")
    parser.add_argument("--backend", default="dwdp", choices=("dwdp", "hf"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute the run CLI."""

    args = build_parser().parse_args(argv)
    config = RuntimeConfig(backend=args.backend, device=args.device)
    runtime = DWDPRuntime.from_pretrained(args.model, config=config)
    tokenizer = getattr(runtime.adapter, "tokenizer", None) if hasattr(runtime, "adapter") else None
    if tokenizer is not None:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        output_ids = runtime.generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        return
    output = runtime.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
