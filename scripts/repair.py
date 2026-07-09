"""CLI: unified repair orchestrator for ARCS failure queues.

Sorts NEGATIVE feedback into per-component piles and runs (or suggests)
the documented repair path for each component. Does not auto-apply DSPy
sidecar prompts to source files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.repair.orchestrator import COMPONENTS, repair_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified repair entry point: extract failure queues, then export / "
            "suggest / optionally run the repair path for ROUTER, SPECIALIST, "
            "VERIFIER, and AMBIGUOUS. DSPy sidecars are never auto-applied."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only: extract in dry-run (unless --skip-extract), suggest commands, do not export/train/run DSPy",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Assume logs/queues/ already exists; skip extract_queues",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=list(COMPONENTS),
        metavar="NAME",
        help=(
            "Limit to one or more components "
            f"({', '.join(COMPONENTS)}). Repeatable."
        ),
    )
    parser.add_argument(
        "--train-router",
        action="store_true",
        help="After exporting labeled ROUTER rows, run python -m arcs.router.train",
    )
    parser.add_argument(
        "--run-dspy",
        action="store_true",
        help="Execute optimize_* scripts instead of only printing instructions",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Specialist domain for DSPy (e.g. MEDICAL). Used with SPECIALIST.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full summary JSON to stdout",
    )
    args = parser.parse_args()

    try:
        summary = repair_all(
            extract_first=not args.skip_extract,
            dry_run=args.dry_run,
            components=args.component,
            train_router=args.train_router,
            run_dspy=args.run_dspy,
            domain=args.domain,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        counts = summary.get("queue_counts") or {}
        print(
            "Queue counts: "
            + ", ".join(f"{k}: {counts.get(k, 0)}" for k in COMPONENTS)
        )
        extract = summary.get("extract") or {}
        if extract.get("skipped"):
            print("Extract: skipped")
        elif extract.get("ok"):
            note = " (dry-run)" if extract.get("dry_run") else ""
            print(f"Extract: ok{note} — {extract.get('summary', '')}")
        else:
            print(f"Extract: failed — {extract.get('error', '')}", file=sys.stderr)

        for component in summary.get("components") or []:
            result = (summary.get("results") or {}).get(component) or {}
            print(f"\n[{component}] queue={result.get('queue_count', 0)}")
            if component == "ROUTER":
                print(f"  rows_exported={result.get('rows_exported', 0)}")
            for note in result.get("notes") or []:
                print(f"  note: {note}")
            for cmd in result.get("suggested") or []:
                print(f"  suggest: {cmd}")
            for item in result.get("ran") or []:
                if isinstance(item, dict) and item.get("command"):
                    status = "ok" if item.get("ok", True) else f"fail/{item.get('returncode')}"
                    print(f"  ran ({status}): {item['command']}")
                elif isinstance(item, dict):
                    print(f"  ran: {item.get('action', item)}")
            for err in result.get("errors") or []:
                print(f"  error: {err}", file=sys.stderr)

        if summary.get("scripts_suggested"):
            print("\nSuggested next commands:")
            for cmd in summary["scripts_suggested"]:
                print(f"  {cmd}")

    if summary.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
