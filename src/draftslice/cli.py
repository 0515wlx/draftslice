"""Command-line entry point for DraftSlice (pre-alpha)."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draftslice",
        description="Exact, constraint-aware draft-head slicing for speculative decoding.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    compile_cmd = sub.add_parser(
        "compile",
        help="Compile a grammar/schema into head plans.",
    )
    compile_cmd.add_argument("--schema", required=True, help="Path to a JSON schema or grammar.")
    compile_cmd.add_argument("--out", required=True, help="Output plan path.")

    bench_cmd = sub.add_parser(
        "bench",
        help="Benchmark draft-head slicing on a target and drafter.",
    )
    bench_cmd.add_argument("--config", required=True, help="Path to a TOML/YAML config.")

    serve_cmd = sub.add_parser(
        "serve",
        help="Run the DraftSlice runtime.",
    )
    serve_cmd.add_argument("--config", required=True, help="Path to a TOML/YAML config.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "compile":
        print("draftslice compile: the plan compiler lands in M1; no plan written.")
        return 2
    if args.command == "bench":
        print("draftslice bench: the benchmark harness lands in M4; no run started.")
        return 2
    if args.command == "serve":
        print("draftslice serve: the runtime lands in M2; no server started.")
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2
