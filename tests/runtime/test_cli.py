from __future__ import annotations

from DWDP.cli.benchmark import build_parser as build_benchmark_parser
from DWDP.cli.run import build_parser as build_run_parser


def test_run_cli_parser() -> None:
    parser = build_run_parser()
    args = parser.parse_args(["--model", "model", "--backend", "dwdp", "--prompt", "Hello"])

    assert args.model == "model"
    assert args.backend == "dwdp"
    assert args.prompt == "Hello"


def test_benchmark_cli_parser() -> None:
    parser = build_benchmark_parser()
    args = parser.parse_args(["--model", "model", "--backend", "hf", "--compare", "dwdp"])

    assert args.model == "model"
    assert args.backend == "hf"
    assert args.compare == "dwdp"
