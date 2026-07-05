from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """Dispatch the installed `dwdp` console command."""

    parser = argparse.ArgumentParser(prog="dwdp")
    parser.add_argument("command", choices=("run", "benchmark", "profile"))
    args, rest = parser.parse_known_args(argv)
    if args.command == "run":
        from .run import main as run_main

        run_main(rest)
    elif args.command == "benchmark":
        from .benchmark import main as benchmark_main

        benchmark_main(rest)
    else:
        from .profile import main as profile_main

        profile_main(rest)


if __name__ == "__main__":
    main()
