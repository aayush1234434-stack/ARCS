"""Synthetic bootstrap for the RQ1 feedback corpus.

RQ1 (attribution-driven repair) normally consumes real NEGATIVE feedback logged
by the demo UI. When that feedback does not yet exist (empty logs/queues), this
script synthesizes an equivalent corpus from artifacts we already have, so the
repair loop can be exercised end-to-end.

It merges three sources and dedupes by query text:

  1. Router misclassifications on the held-out test set
     (artifacts/eval-results/misclassified_test.json). Every row is a routing
     error, so it is attributed to ROUTER directly.
  2. Baseline pipeline eval rows where the router sent the query to the wrong
     domain (predicted_domain != expected_domain). Also attributed to ROUTER.
  3. Baseline pipeline eval rows that were correctly routed but still FAILed
     (status == FAIL and predicted_domain == expected_domain). These are run
     through arcs.post.attribution.attribute() on a synthetic state, so the
     blame lands on SPECIALIST / VERIFIER / AMBIGUOUS as the rule set decides.

Each output record (one JSON object per line) looks like:

    {
      "query_id": "rq1-001",
      "query": "...",
      "correct_domain": "LEGAL",
      "expected_domain": "LEGAL",
      "user_feedback": "NEGATIVE",
      "source": "misclassified_test|eval_misroute|eval_fail",
      "route": {"domain": "...", "confidence": 0.9},
      "verification": {"verdict": "FAIL", "score": 0.5},
      "attribution": {"component": "ROUTER", "rule": 3, "reason": "..."}
    }

Usage:
    python scripts/bootstrap_rq1_corpus.py --dry-run
    python scripts/bootstrap_rq1_corpus.py --output data/rq1/feedback_corpus.jsonl
    python scripts/bootstrap_rq1_corpus.py --real-only
    python scripts/bootstrap_rq1_corpus.py --real-only --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.post.attribution import attribute

DEFAULT_MISCLASSIFIED = config.EVAL_RESULTS_DIR / "misclassified_test.json"
DEFAULT_EXPERIMENT = (
    config.EXPERIMENTS_DIR
    / "2026-07-10T07-24-20_baseline-v1-full-pipeline"
    / "experiment.json"
)
DEFAULT_OUTPUT = config.DATA_DIR / "rq1" / "feedback_corpus.jsonl"
DEFAULT_REAL_OUTPUT = config.DATA_DIR / "rq1" / "feedback_corpus_real.jsonl"
REQUESTS_LOG = config.LOGS_DIR / "requests.jsonl"

MIN_ROUTER_EXAMPLES = 8
RQ1_V2_MIN_NEGATIVES = 40
RQ1_V2_MIN_ROUTER = 15


def _norm_query(query: str) -> str:
    """Normalization key for dedupe (case- and whitespace-insensitive)."""
    return " ".join(str(query).strip().lower().split())


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse {path}: {exc}", file=sys.stderr)
        return None


def _records_from_misclassified(path: Path) -> list[dict[str, Any]]:
    """Source 1: every router misclassification is a ROUTER failure."""
    data = _load_json(path)
    if not isinstance(data, list):
        return []

    records: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        query = row.get("text")
        true_label = row.get("true_label")
        predicted = row.get("predicted_label")
        if not query or not true_label or not predicted:
            continue
        records.append(
            {
                "query": query,
                "correct_domain": true_label,
                "expected_domain": true_label,
                "user_feedback": "NEGATIVE",
                "source": "misclassified_test",
                "route": {"domain": predicted, "confidence": row.get("confidence")},
                "verification": {},
                "attribution": {
                    "component": "ROUTER",
                    "rule": 3,
                    "reason": "Router misclassified query on the held-out test set.",
                },
            }
        )
    return records


def _experiment_rows(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        return []
    rows = data.get("meta", {}).get("rows", [])
    return [r for r in rows if isinstance(r, dict)]


def _records_from_misroutes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Source 2: eval rows routed to the wrong domain -> ROUTER."""
    records: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected_domain")
        predicted = row.get("predicted_domain")
        query = row.get("query")
        if not query or not expected or not predicted:
            continue
        if predicted == expected:
            continue
        records.append(
            {
                "query": query,
                "correct_domain": expected,
                "expected_domain": expected,
                "user_feedback": "NEGATIVE",
                "source": "eval_misroute",
                "route": {
                    "domain": predicted,
                    "confidence": row.get("router_confidence"),
                },
                "verification": row.get("verification") or {},
                "attribution": {
                    "component": "ROUTER",
                    "rule": 3,
                    "reason": "Router routed eval query to the wrong domain.",
                },
            }
        )
    return records


def _synthetic_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a verification block including verification_type.

    The stored eval rows keep only ``verdict``/``score``; the attribution rule
    set needs ``verification_type`` (SANDBOX vs LLM_JUDGE) to fire rule 1
    correctly, so we recover it from the row's ``verifier`` field.
    """
    verification = dict(row.get("verification") or {})
    verifier = str(row.get("verifier") or "").lower()
    if "verification_type" not in verification:
        if verifier == "sandbox":
            verification["verification_type"] = "SANDBOX"
        elif verifier in {"llm_judge", "judge"}:
            verification["verification_type"] = "LLM_JUDGE"
    return verification


def _records_from_fails(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Source 3: correctly-routed FAILs -> attribute() decides the component."""
    records: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected_domain")
        predicted = row.get("predicted_domain")
        query = row.get("query")
        if not query or not expected or not predicted:
            continue
        if row.get("status") != "FAIL" or predicted != expected:
            continue

        verification = _synthetic_verification(row)
        route = {"domain": predicted, "confidence": row.get("router_confidence")}
        synthetic_state = {
            "route": route,
            "verification": verification,
            "user_feedback": "NEGATIVE",
        }
        attribution = attribute(synthetic_state)

        records.append(
            {
                "query": query,
                "correct_domain": expected,
                "expected_domain": expected,
                "user_feedback": "NEGATIVE",
                "source": "eval_fail",
                "route": route,
                "verification": verification,
                "attribution": attribution,
            }
        )
    return records


def build_corpus(
    *,
    misclassified_path: Path,
    experiment_path: Path,
) -> list[dict[str, Any]]:
    """Merge all sources (order 1->2->3) and dedupe by normalized query text."""
    rows = _experiment_rows(experiment_path)
    ordered = (
        _records_from_misclassified(misclassified_path)
        + _records_from_misroutes(rows)
        + _records_from_fails(rows)
    )

    seen: set[str] = set()
    corpus: list[dict[str, Any]] = []
    for record in ordered:
        key = _norm_query(record["query"])
        if not key or key in seen:
            continue
        seen.add(key)
        corpus.append(record)

    for index, record in enumerate(corpus, start=1):
        record["query_id"] = f"rq1-{index:03d}"

    # query_id first for readability.
    return [
        {"query_id": r.pop("query_id"), **r} for r in corpus
    ]


def _user_feedback(record: dict[str, Any]) -> str | None:
    feedback = record.get("user_feedback")
    if feedback is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            feedback = metadata.get("user_feedback")
    if feedback is None or not isinstance(feedback, str):
        return None
    return feedback.strip().upper()


def build_real_corpus(*, requests_path: Path) -> list[dict[str, Any]]:
    """Build corpus from live demo/CLI feedback in logs/requests.jsonl only."""
    if not requests_path.exists():
        return []

    # Keep the last NEGATIVE record per normalized query (newest wins).
    by_query: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    with requests_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if _user_feedback(record) != "NEGATIVE":
                continue

            query = str(record.get("query") or "").strip()
            if not query:
                continue

            route = record.get("route") if isinstance(record.get("route"), dict) else {}
            verification = (
                record.get("verification")
                if isinstance(record.get("verification"), dict)
                else {}
            )
            attribution = record.get("attribution")
            if not isinstance(attribution, dict) or not attribution.get("component"):
                attribution = attribute(
                    {
                        "route": route,
                        "verification": verification,
                        "user_feedback": "NEGATIVE",
                        "specialist": record.get("specialist"),
                    }
                )

            correct_domain = record.get("correct_domain") or record.get("expected_domain")
            if isinstance(correct_domain, str):
                correct_domain = correct_domain.strip().upper()
            else:
                correct_domain = None

            expected = correct_domain or (
                str(route.get("domain")).strip().upper() if route.get("domain") else None
            )

            corpus_row: dict[str, Any] = {
                "query_id": str(record.get("query_id") or ""),
                "query": query,
                "correct_domain": correct_domain or expected,
                "expected_domain": expected,
                "user_feedback": "NEGATIVE",
                "source": "demo_feedback",
                "route": route,
                "verification": verification,
                "attribution": attribution,
            }

            key = _norm_query(query)
            if key not in by_query:
                order.append(key)
            by_query[key] = corpus_row

    corpus: list[dict[str, Any]] = []
    for index, key in enumerate(order, start=1):
        row = by_query[key]
        if not row.get("query_id"):
            row["query_id"] = f"rq1-real-{index:03d}"
        corpus.append(row)
    return corpus


def _print_real_summary(corpus: list[dict[str, Any]]) -> None:
    by_component = Counter(r["attribution"]["component"] for r in corpus)
    router = by_component.get("ROUTER", 0)
    total = len(corpus)
    with_domain = sum(
        1
        for r in corpus
        if isinstance(r.get("correct_domain"), str) and str(r["correct_domain"]).strip()
    )

    print("RQ1 real-feedback corpus summary")
    print(f"  source:            {REQUESTS_LOG}")
    print(f"  total negatives:   {total}")
    print(f"  with correct_domain: {with_domain}")
    print("  by attribution component:")
    for component in ("ROUTER", "SPECIALIST", "VERIFIER", "AMBIGUOUS"):
        print(f"    {component:11s} {by_component.get(component, 0)}")
    print(f"  ROUTER count:      {router}")
    print(f"  non-ROUTER count:  {total - router}")
    print()
    print(f"  RQ1 v2 thresholds: ≥{RQ1_V2_MIN_NEGATIVES} negatives, ≥{RQ1_V2_MIN_ROUTER} ROUTER")
    ready = total >= RQ1_V2_MIN_NEGATIVES and router >= RQ1_V2_MIN_ROUTER
    print(f"  RQ1 v2 ready:      {'yes' if ready else 'no'}")


def _print_summary(corpus: list[dict[str, Any]]) -> None:
    by_component = Counter(r["attribution"]["component"] for r in corpus)
    by_source = Counter(r["source"] for r in corpus)
    router = by_component.get("ROUTER", 0)
    total = len(corpus)

    print("RQ1 feedback corpus summary")
    print(f"  total negatives: {total}")
    print("  by attribution component:")
    for component in ("ROUTER", "SPECIALIST", "VERIFIER", "AMBIGUOUS"):
        print(f"    {component:11s} {by_component.get(component, 0)}")
    print("  by source:")
    for source in ("misclassified_test", "eval_misroute", "eval_fail"):
        print(f"    {source:19s} {by_source.get(source, 0)}")
    print(f"  ROUTER count:     {router}")
    print(f"  non-ROUTER count: {total - router}")

    if router < MIN_ROUTER_EXAMPLES:
        print(
            f"\nWarning: only {router} ROUTER example(s) "
            f"(< {MIN_ROUTER_EXAMPLES}); router retraining data may be too thin.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap a synthetic RQ1 feedback corpus from router-eval "
            "misclassifications and baseline pipeline failures."
        ),
    )
    parser.add_argument(
        "--misclassified",
        type=Path,
        default=DEFAULT_MISCLASSIFIED,
        help=f"Router misclassified_test.json (default: {DEFAULT_MISCLASSIFIED})",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_EXPERIMENT,
        help=f"Baseline pipeline experiment.json (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help=(
            "Build ONLY from logs/requests.jsonl (live demo/CLI feedback). "
            "Excludes eval export and misclassified_test.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path (default: feedback_corpus_real.jsonl with "
            "--real-only, else feedback_corpus.jsonl)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts by attribution without writing the corpus.",
    )
    args = parser.parse_args()

    output = args.output or (
        DEFAULT_REAL_OUTPUT if args.real_only else DEFAULT_OUTPUT
    )

    if args.real_only:
        if not REQUESTS_LOG.exists():
            print(f"Error: requests log not found: {REQUESTS_LOG}", file=sys.stderr)
            sys.exit(1)
        corpus = build_real_corpus(requests_path=REQUESTS_LOG)
        if not corpus:
            print(
                "Error: no NEGATIVE feedback rows in requests.jsonl.",
                file=sys.stderr,
            )
            sys.exit(1)
        _print_real_summary(corpus)
    else:
        corpus = build_corpus(
            misclassified_path=args.misclassified,
            experiment_path=args.experiment,
        )
        if not corpus:
            print(
                "Error: no records built — check that source files exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        _print_summary(corpus)

    if args.dry_run:
        print("\n(dry-run: no file written)")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in corpus:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(corpus)} record(s) to {output}")


if __name__ == "__main__":
    main()
