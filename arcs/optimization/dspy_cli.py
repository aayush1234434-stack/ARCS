"""Shared CLI helpers for optimize_*.py scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from arcs.optimization.dspy_common import (
    GROQ_COPRO_DEFAULT_BREADTH,
    GROQ_COPRO_DEFAULT_DEPTH,
    ensure_sidecar_written,
    validate_copro_breadth,
)


def add_max_examples_arg(parser: argparse.ArgumentParser, *, default: int = 20) -> None:
    parser.add_argument(
        "--max-examples",
        "--limit",
        type=int,
        default=default,
        metavar="N",
        dest="max_examples",
        help=f"Max examples to load from the queue (default: {default})",
    )


def add_groq_copro_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--breadth",
        type=int,
        default=GROQ_COPRO_DEFAULT_BREADTH,
        help=(
            "COPRO breadth (default: "
            f"{GROQ_COPRO_DEFAULT_BREADTH}; must be >1; Groq API n=1 per call)"
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=GROQ_COPRO_DEFAULT_DEPTH,
        help=f"COPRO depth (default: {GROQ_COPRO_DEFAULT_DEPTH})",
    )


def add_judge_copro_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="COPRO breadth (default: 5; NVIDIA judge LM, not Groq)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=GROQ_COPRO_DEFAULT_DEPTH,
        help=f"COPRO depth (default: {GROQ_COPRO_DEFAULT_DEPTH})",
    )


def validate_optimize_args(args: argparse.Namespace) -> None:
    if args.max_examples < 1:
        raise ValueError("--max-examples/--limit must be >= 1")
    validate_copro_breadth(args.breadth)


def finalize_optimize_run(summary: dict, output_path: Path) -> None:
    ensure_sidecar_written(summary, output_path)
