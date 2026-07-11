"""CLI: experimental DSPy optimization for the specification generator.

Writes a sidecar under artifacts/prompts/ — does not overwrite spec_generator.py.
Only uses queue rows with incomplete specs (few required_elements or
non-empty missing_required_elements).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.optimization.dspy_cli import (
    add_groq_copro_args,
    add_max_examples_arg,
    finalize_optimize_run,
    validate_optimize_args,
)
from arcs.optimization.dspy_spec import (
    DEFAULT_OUTPUT,
    DEFAULT_SPECIALIST_QUEUE,
    DEFAULT_VERIFIER_QUEUE,
    optimize_spec_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EXPERIMENTAL: optimize the spec-generator system prompt with DSPy "
            "COPRO using incomplete-spec rows from specialist/verifier queues. "
            "Writes a sidecar for human review — does not modify "
            "spec_generator.py."
        ),
    )
    parser.add_argument(
        "--specialist-queue",
        type=Path,
        default=DEFAULT_SPECIALIST_QUEUE,
        help=f"specialist_queue.jsonl path (default: {DEFAULT_SPECIALIST_QUEUE})",
    )
    parser.add_argument(
        "--verifier-queue",
        type=Path,
        default=DEFAULT_VERIFIER_QUEUE,
        help=f"verifier_queue.jsonl path (default: {DEFAULT_VERIFIER_QUEUE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sidecar prompt path (default: {DEFAULT_OUTPUT})",
    )
    add_max_examples_arg(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List incomplete-spec examples; do not call DSPy or write files",
    )
    add_groq_copro_args(parser)
    args = parser.parse_args()

    try:
        validate_optimize_args(args)
        summary = optimize_spec_prompt(
            specialist_queue=args.specialist_queue,
            verifier_queue=args.verifier_queue,
            output_path=args.output,
            max_examples=args.max_examples,
            dry_run=args.dry_run,
            breadth=args.breadth,
            depth=args.depth,
        )
        finalize_optimize_run(summary, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"examples={summary['examples']} dry_run={summary['dry_run']} "
        f"written={summary['written']} experimental={summary.get('experimental')} "
        f"output={summary['output']}"
    )


if __name__ == "__main__":
    main()
