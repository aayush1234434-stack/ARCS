"""CLI: optimize MEDICAL specialist system prompt with DSPy (COPRO).

Writes a sidecar file under artifacts/prompts/ — does not overwrite medical.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.optimization.dspy_medical import (
    DEFAULT_OUTPUT,
    DEFAULT_QUEUE,
    optimize_medical_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the MEDICAL specialist system prompt with DSPy COPRO "
            "using specialist_queue.jsonl failures. Writes a sidecar prompt "
            "file for human review — does not modify medical.py."
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help=f"specialist_queue.jsonl path (default: {DEFAULT_QUEUE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sidecar prompt path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        metavar="N",
        help="Max MEDICAL examples to load from the queue (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List examples that would be used; do not call DSPy or write files",
    )
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="COPRO breadth (prompt candidates per round; default: 5)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="COPRO depth (optimization rounds; default: 2)",
    )
    args = parser.parse_args()

    if args.max_examples < 1:
        print("Error: --max-examples must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        summary = optimize_medical_prompt(
            queue_path=args.queue,
            output_path=args.output,
            max_examples=args.max_examples,
            dry_run=args.dry_run,
            breadth=args.breadth,
            depth=args.depth,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"examples={summary['examples']} dry_run={summary['dry_run']} "
        f"written={summary['written']} output={summary['output']}"
    )


if __name__ == "__main__":
    main()
