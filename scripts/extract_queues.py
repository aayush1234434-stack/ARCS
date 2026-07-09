"""CLI: extract per-component repair queues from request logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/extract_queues.py`` from project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.post.queues import extract_queues, format_summary, queue_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter NEGATIVE feedback from logs/requests.jsonl into "
            "per-component queue files (router, specialist, verifier, ambiguous)."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to requests.jsonl (default: logs/requests.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for queue JSONL files (default: logs/queues/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write queue files",
    )
    args = parser.parse_args()

    try:
        queues = extract_queues(
            input_path=args.input,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    counts = queue_counts(queues)
    print(format_summary(counts))
    if args.dry_run:
        print("(dry-run: no files written)", file=sys.stderr)
    else:
        out = args.output_dir or (_ROOT / "logs" / "queues")
        print(f"Wrote queues to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
