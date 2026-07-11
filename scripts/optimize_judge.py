"""CLI: optimize LLM judge system prompt with DSPy (COPRO).

Writes a sidecar file under artifacts/prompts/ — does not overwrite judge.py.
Sandbox verifier path is out of scope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.optimization.dspy_cli import (
    add_judge_copro_args,
    add_max_examples_arg,
    finalize_optimize_run,
    validate_optimize_args,
)
from arcs.optimization.dspy_judge import (
    DEFAULT_OUTPUT,
    DEFAULT_QUEUE,
    optimize_judge_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the LLM judge system prompt with DSPy COPRO using "
            "verifier_queue.jsonl false-PASS failures. Writes a sidecar "
            "prompt for human review — does not modify judge.py. "
            "Uses NVIDIA API (not Groq)."
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help=f"verifier_queue.jsonl path (default: {DEFAULT_QUEUE})",
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
        help="List examples that would be used; do not call DSPy or write files",
    )
    add_judge_copro_args(parser)
    args = parser.parse_args()

    try:
        validate_optimize_args(args)
        summary = optimize_judge_prompt(
            queue_path=args.queue,
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
        f"written={summary['written']} output={summary['output']}"
    )


if __name__ == "__main__":
    main()
