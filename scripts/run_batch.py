"""CLI: run a batch of queries through the ARCS pipeline for log collection.

Data collection only — does not retrain models or run DSPy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as ``python scripts/run_batch.py`` from project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config, progress
from arcs.main import run_pipeline
from arcs.post import feedback, logger

VALID_DOMAINS = frozenset({"CODING", "MEDICAL", "LEGAL", "GENERAL"})
VALID_FEEDBACK = frozenset({"POSITIVE", "NEGATIVE"})

DEFAULT_INPUT = config.DATA_DIR / "batch_queries.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(row, dict):
                print(
                    f"Warning: skipping non-object on line {line_number}",
                    file=sys.stderr,
                )
                continue
            query = row.get("query")
            if not isinstance(query, str) or not query.strip():
                print(
                    f"Warning: skipping line {line_number} — missing query",
                    file=sys.stderr,
                )
                continue
            rows.append(row)
    return rows


def _normalize_feedback(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in {"", "NULL", "NONE"}:
        return None
    if normalized not in VALID_FEEDBACK:
        print(
            f"Warning: ignoring invalid feedback {value!r}",
            file=sys.stderr,
        )
        return None
    return normalized


def _normalize_domain(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized not in VALID_DOMAINS:
        print(
            f"Warning: ignoring invalid expected_domain {value!r}",
            file=sys.stderr,
        )
        return None
    return normalized


def _preview(query: str, limit: int = 72) -> str:
    text = query.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def run_batch(
    rows: list[dict[str, Any]],
    *,
    apply_feedback: bool,
    dry_run: bool,
) -> dict[str, int]:
    """Run each row through the pipeline. Returns summary counts."""
    summary = {
        "total": 0,
        "ok": 0,
        "error": 0,
        "pass": 0,
        "fail": 0,
        "unknown": 0,
        "feedback_positive": 0,
        "feedback_negative": 0,
        "feedback_none": 0,
    }

    for index, row in enumerate(rows, start=1):
        query = str(row["query"]).strip()
        expected_domain = _normalize_domain(row.get("expected_domain"))
        signal = _normalize_feedback(row.get("feedback")) if apply_feedback else None

        summary["total"] += 1
        print(
            f"[{index}/{len(rows)}] {_preview(query)}"
            + (f"  feedback={signal}" if signal else "")
            + (f"  expected={expected_domain}" if expected_domain else ""),
            file=sys.stderr,
        )

        if dry_run:
            summary["ok"] += 1
            if signal == "POSITIVE":
                summary["feedback_positive"] += 1
            elif signal == "NEGATIVE":
                summary["feedback_negative"] += 1
            else:
                summary["feedback_none"] += 1
            continue

        try:
            state = run_pipeline(query)
        except Exception as exc:
            summary["error"] += 1
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        feedback_signal = feedback.collect(explicit=signal) if signal else None
        if feedback_signal is None:
            summary["feedback_none"] += 1
        elif feedback_signal["user_feedback"] == "POSITIVE":
            summary["feedback_positive"] += 1
        else:
            summary["feedback_negative"] += 1

        record = feedback.apply(state, feedback_signal)
        if expected_domain is not None:
            record["expected_domain"] = expected_domain

        logger.log(record)

        status = str(record.get("status", "UNKNOWN")).upper()
        if status == "PASS":
            summary["pass"] += 1
        elif status == "FAIL":
            summary["fail"] += 1
        else:
            summary["unknown"] += 1
        summary["ok"] += 1

        domain = (record.get("route") or {}).get("domain", "?")
        print(
            f"  status={status} domain={domain} "
            f"verdict={(record.get('verification') or {}).get('verdict')}",
            file=sys.stderr,
        )

    return summary


def _print_summary(summary: dict[str, int], *, dry_run: bool) -> None:
    print("", file=sys.stderr)
    print("=== Batch summary ===", file=sys.stderr)
    print(f"total:    {summary['total']}", file=sys.stderr)
    print(f"ok:       {summary['ok']}", file=sys.stderr)
    print(f"error:    {summary['error']}", file=sys.stderr)
    if not dry_run:
        print(f"pass:     {summary['pass']}", file=sys.stderr)
        print(f"fail:     {summary['fail']}", file=sys.stderr)
        print(f"unknown:  {summary['unknown']}", file=sys.stderr)
    print(f"feedback+: {summary['feedback_positive']}", file=sys.stderr)
    print(f"feedback-: {summary['feedback_negative']}", file=sys.stderr)
    print(f"feedback0: {summary['feedback_none']}", file=sys.stderr)
    if dry_run:
        print("(dry-run: no pipeline calls, no logs written)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch queries through ARCS to collect logs for retraining. "
            "Does not retrain or run DSPy."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Batch JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without calling the pipeline or writing logs",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Ignore feedback fields in the batch file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress pipeline progress messages (batch progress still prints)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: batch file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = _load_rows(args.input)
    if args.limit is not None:
        if args.limit < 0:
            print("Error: --limit must be >= 0", file=sys.stderr)
            sys.exit(1)
        rows = rows[: args.limit]

    if not rows:
        print("Error: no valid rows to run", file=sys.stderr)
        sys.exit(1)

    progress.set_verbose(not args.quiet and not args.dry_run)

    print(
        f"Loaded {len(rows)} row(s) from {args.input}"
        + (" [dry-run]" if args.dry_run else "")
        + (" [no-feedback]" if args.no_feedback else ""),
        file=sys.stderr,
    )

    summary = run_batch(
        rows,
        apply_feedback=not args.no_feedback,
        dry_run=args.dry_run,
    )
    _print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
