"""CLI: extract queues → export router examples → retrain DistilBERT.

Does not run DSPy. Requires labeled router-queue rows
(expected_domain or correct_domain).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.post.queues import extract_queues, format_summary, queue_counts
from arcs.router.export_training import DEFAULT_CSV, DEFAULT_QUEUE, export_router_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Close the router repair loop: extract queues, export labeled "
            "ROUTER failures into router_train.csv, then retrain DistilBERT."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Assume logs/queues/ already exists; skip extract_queues",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Export CSV only; do not run training",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help=f"Router queue path (default: {DEFAULT_QUEUE})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Training CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite output CSV with only newly exported rows",
    )
    args = parser.parse_args()

    if not args.skip_extract:
        print("Step 1/3: extract queues from logs/requests.jsonl", file=sys.stderr)
        try:
            queues = extract_queues()
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(format_summary(queue_counts(queues)), file=sys.stderr)
    else:
        print("Step 1/3: skipped (--skip-extract)", file=sys.stderr)

    print("Step 2/3: export labeled router-queue rows → training CSV", file=sys.stderr)
    try:
        written = export_router_examples(
            queue_path=args.queue,
            output_csv=args.output_csv,
            append=not args.no_append,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Hint: run extract_queues, then label rows with "
            "scripts/label_router_failures.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Wrote {written} new training row(s) to {args.output_csv}", file=sys.stderr)
    if written == 0:
        print(
            "No new labeled ROUTER examples to add. "
            "Label failures first:\n"
            "  python scripts/label_router_failures.py --list\n"
            "  python scripts/label_router_failures.py --interactive",
            file=sys.stderr,
        )
        if args.skip_train:
            sys.exit(0)
        print("Skipping train because there is nothing new to learn.", file=sys.stderr)
        sys.exit(0)

    if args.skip_train:
        print("Step 3/3: skipped (--skip-train)", file=sys.stderr)
        print("retrain complete (export only)")
        print(f"CSV: {args.output_csv}")
        return

    print("Step 3/3: retrain DistilBERT router", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, "-m", "arcs.router.train"],
        cwd=str(config.PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(
            f"Error: training failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print("retrain complete")
    print(f"Model: {config.ROUTER_MODEL_DIR}")


if __name__ == "__main__":
    main()
