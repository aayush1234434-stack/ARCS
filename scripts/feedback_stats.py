"""Print feedback collection stats for Phase 3 / RQ1 v2 readiness.

Counts POSITIVE/NEGATIVE signals in logs/requests.jsonl and
logs/eval_failures.jsonl, per-component queue totals, and whether the
corpus meets RQ1 v2 thresholds (≥40 negatives, ≥15 ROUTER-attributed).

Usage:
    python scripts/feedback_stats.py
    python scripts/feedback_stats.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.post.queues import (
    COMPONENTS,
    EVAL_FAILURES_LOG,
    REQUESTS_LOG,
    _attribution_component,
    _read_records,
    _user_feedback,
    extract_queues,
    format_summary,
    queue_counts,
)

RQ1_V2_MIN_NEGATIVES = 40
RQ1_V2_MIN_ROUTER = 15


def _load_all_feedback_records(*, include_eval_failures: bool = True) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if REQUESTS_LOG.exists():
        paths.append(REQUESTS_LOG)
    if include_eval_failures and EVAL_FAILURES_LOG.exists():
        paths.append(EVAL_FAILURES_LOG)
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_read_records(path))
    return records


def _feedback_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "OTHER": 0, "NONE": 0}
    for record in records:
        signal = _user_feedback(record)
        if signal == "POSITIVE":
            counts["POSITIVE"] += 1
        elif signal == "NEGATIVE":
            counts["NEGATIVE"] += 1
        elif signal is None:
            counts["NONE"] += 1
        else:
            counts["OTHER"] += 1
    return counts


def _attribution_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {component: 0 for component in COMPONENTS}
    for record in records:
        if _user_feedback(record) != "NEGATIVE":
            continue
        component = _attribution_component(record)
        if component in counts:
            counts[component] += 1
    return counts


def _negatives_with_correct_domain(records: list[dict[str, Any]]) -> int:
    n = 0
    for record in records:
        if _user_feedback(record) != "NEGATIVE":
            continue
        domain = record.get("correct_domain") or record.get("expected_domain")
        if isinstance(domain, str) and domain.strip():
            n += 1
    return n


def gather_stats(*, include_eval_failures: bool = True) -> dict[str, Any]:
    paths: list[Path] = []
    if REQUESTS_LOG.exists():
        paths.append(REQUESTS_LOG)
    if include_eval_failures and EVAL_FAILURES_LOG.exists():
        paths.append(EVAL_FAILURES_LOG)

    records = _load_all_feedback_records(include_eval_failures=include_eval_failures)
    feedback = _feedback_counts(records)
    attribution = _attribution_counts(records)
    negatives = feedback["NEGATIVE"]
    router_negatives = attribution.get("ROUTER", 0)

    try:
        queues = extract_queues(dry_run=True, include_eval_failures=include_eval_failures)
        queue_totals = queue_counts(queues)
    except FileNotFoundError:
        queue_totals = {component: 0 for component in COMPONENTS}

    rq1_ready = (
        negatives >= RQ1_V2_MIN_NEGATIVES and router_negatives >= RQ1_V2_MIN_ROUTER
    )

    return {
        "sources": [str(p) for p in paths],
        "total_records": len(records),
        "feedback": feedback,
        "negatives_with_domain_label": _negatives_with_correct_domain(records),
        "attribution_on_negatives": attribution,
        "queue_counts": queue_totals,
        "rq1_v2": {
            "ready": rq1_ready,
            "min_negatives": RQ1_V2_MIN_NEGATIVES,
            "min_router_attributed": RQ1_V2_MIN_ROUTER,
            "negatives": negatives,
            "router_attributed": router_negatives,
        },
    }


def _print_human(stats: dict[str, Any]) -> None:
    print("=== Feedback collection stats ===")
    print(f"sources: {', '.join(stats.get('sources') or []) or '(none)'}")
    print(f"total records: {stats.get('total_records', 0)}")
    fb = stats.get("feedback") or {}
    print(
        f"signals: POSITIVE={fb.get('POSITIVE', 0)}  "
        f"NEGATIVE={fb.get('NEGATIVE', 0)}  "
        f"none={fb.get('NONE', 0)}"
    )
    print(f"negatives with correct_domain label: {stats.get('negatives_with_domain_label', 0)}")
    print()
    print("Attribution on NEGATIVE rows:")
    attr = stats.get("attribution_on_negatives") or {}
    for component in COMPONENTS:
        print(f"  {component}: {attr.get(component, 0)}")
    print()
    print("Queue counts (extract dry-run):")
    qc = stats.get("queue_counts") or {}
    print(f"  {format_summary(qc)}")
    print()
    rq1 = stats.get("rq1_v2") or {}
    ready = "yes" if rq1.get("ready") else "no"
    print(f"RQ1 v2 ready: {ready}")
    print(
        f"  negatives: {rq1.get('negatives', 0)} / {rq1.get('min_negatives', RQ1_V2_MIN_NEGATIVES)}"
    )
    print(
        f"  ROUTER-attributed: {rq1.get('router_attributed', 0)} / "
        f"{rq1.get('min_router_attributed', RQ1_V2_MIN_ROUTER)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback stats and RQ1 v2 readiness.")
    parser.add_argument(
        "--requests-only",
        action="store_true",
        help="Count logs/requests.jsonl only (ignore eval_failures.jsonl)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    stats = gather_stats(include_eval_failures=not args.requests_only)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        _print_human(stats)

    rq1 = stats.get("rq1_v2") or {}
    if not rq1.get("ready"):
        sys.exit(1)


if __name__ == "__main__":
    main()
