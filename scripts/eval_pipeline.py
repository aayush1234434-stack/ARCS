"""CLI: run the held-out eval set through ARCS and save experiment artifacts.

Does not write to logs/requests.jsonl and does not collect feedback/attribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config, progress
from arcs.eval.experiments import save_experiment
from arcs.eval.metrics import (
    VALID_DOMAINS,
    aggregate_experiment,
    pipeline_summary,
    router_accuracy,
)
from arcs.main import run_pipeline

DEFAULT_INPUT = config.DATA_DIR / "eval_queries.jsonl"
DOMAIN_SET = frozenset(VALID_DOMAINS)


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


def _normalize_domain(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized not in DOMAIN_SET:
        return None
    return normalized


def _derive_status(state: dict[str, Any] | None, *, error: str | None) -> str:
    if error:
        return "ERROR"
    if not isinstance(state, dict):
        return "UNKNOWN"
    verdict = (state.get("verification") or {}).get("verdict")
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def _build_row_result(
    row: dict[str, Any],
    state: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    expected = _normalize_domain(row.get("expected_domain"))
    result: dict[str, Any] = {
        "id": row.get("id"),
        "query": str(row.get("query") or "").strip(),
        "expected_domain": expected,
    }

    if error:
        result["status"] = "ERROR"
        result["error"] = error
        result["predicted_domain"] = None
        result["router_confidence"] = None
        result["use_fallback"] = None
        result["verification"] = {"verdict": None, "score": None}
        result["pipeline_id"] = None
        result["verifier"] = None
        result["timing"] = {}
        return result

    assert state is not None
    route = state.get("route") or {}
    pipeline = state.get("pipeline") or {}
    verification = state.get("verification") or {}
    timing = state.get("timing") or {}

    predicted = route.get("domain") if isinstance(route, dict) else None
    confidence = route.get("confidence") if isinstance(route, dict) else None
    try:
        confidence_f = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_f = None

    score = verification.get("score") if isinstance(verification, dict) else None
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None

    result["predicted_domain"] = (
        str(predicted).strip().upper() if predicted is not None else None
    )
    result["router_confidence"] = confidence_f
    result["use_fallback"] = (
        bool(route.get("use_fallback")) if isinstance(route, dict) else None
    )
    result["status"] = _derive_status(state, error=None)
    result["verification"] = {
        "verdict": verification.get("verdict") if isinstance(verification, dict) else None,
        "score": score_f,
    }
    result["pipeline_id"] = (
        pipeline.get("pipeline_id") if isinstance(pipeline, dict) else None
    )
    result["verifier"] = pipeline.get("verifier") if isinstance(pipeline, dict) else None
    result["timing"] = dict(timing) if isinstance(timing, dict) else {}
    result["error"] = None
    return result


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    ids: set[str] | None,
    domains: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if ids is not None:
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id not in ids:
                continue
        if domains is not None:
            domain = _normalize_domain(row.get("expected_domain"))
            if domain not in domains:
                continue
        filtered.append(row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _parse_csv_set(raw: str | None, *, label: str) -> set[str] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} filter is empty")
    return set(parts)


def _print_progress(
    index: int,
    total: int,
    result: dict[str, Any],
) -> None:
    row_id = result.get("id") or "?"
    status = result.get("status") or "?"
    routed = result.get("predicted_domain") or "?"
    expected = result.get("expected_domain") or "?"
    timing = result.get("timing") or {}
    total_ms = timing.get("total_ms") if isinstance(timing, dict) else None
    latency = f"  {int(total_ms)}ms" if isinstance(total_ms, (int, float)) else ""
    err = result.get("error")
    extra = f"  error={err}" if err else ""
    print(
        f"[{index}/{total}] {row_id}  status={status}  "
        f"routed={routed}  expected={expected}{latency}{extra}",
        file=sys.stderr,
    )


def _print_summary(experiment: dict[str, Any], *, dry_run: bool, saved_to: Path | None) -> None:
    print("", file=sys.stderr)
    print("=== Pipeline eval summary ===", file=sys.stderr)
    meta = experiment.get("meta") or {}
    print(f"name:     {experiment.get('name')}", file=sys.stderr)
    if meta.get("run_id"):
        print(f"run_id:   {meta['run_id']}", file=sys.stderr)

    if dry_run:
        print(f"planned:  {meta.get('n_planned', 0)}", file=sys.stderr)
        counts = meta.get("planned_domain_counts") or {}
        for domain in VALID_DOMAINS:
            print(f"  {domain}: {counts.get(domain, 0)}", file=sys.stderr)
        print("(dry-run: no pipeline calls, no artifacts written)", file=sys.stderr)
        return

    router = experiment.get("router") or {}
    pipeline = experiment.get("pipeline") or {}
    acc = router.get("accuracy")
    print(
        f"router:   n={router.get('n')}  "
        f"accuracy={acc if acc is None else f'{acc:.3f}'}",
        file=sys.stderr,
    )
    print(
        f"pipeline: n={pipeline.get('n')}  "
        f"PASS={pipeline.get('status_counts', {}).get('PASS', 0)}  "
        f"FAIL={pipeline.get('status_counts', {}).get('FAIL', 0)}  "
        f"UNKNOWN={pipeline.get('status_counts', {}).get('UNKNOWN', 0)}  "
        f"errors={pipeline.get('error_count', 0)}",
        file=sys.stderr,
    )
    latency = (pipeline.get("latency_ms") or {}).get("total_ms") or {}
    if latency.get("count"):
        print(
            f"latency:  mean={latency.get('mean')}ms  "
            f"p50={latency.get('p50')}ms  p95={latency.get('p95')}ms",
            file=sys.stderr,
        )
    if saved_to is not None:
        print(f"saved:    {saved_to}", file=sys.stderr)


def run_eval(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run eval rows. Returns (per-row results, counters)."""
    counters = {
        "total": 0,
        "ok": 0,
        "errors": 0,
        "pass": 0,
        "fail": 0,
        "unknown": 0,
    }
    results: list[dict[str, Any]] = []
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        counters["total"] += 1
        row_id = row.get("id") or f"row-{index}"
        expected = _normalize_domain(row.get("expected_domain"))

        if dry_run:
            result = {
                "id": row.get("id"),
                "query": str(row.get("query") or "").strip(),
                "expected_domain": expected,
                "status": "PLANNED",
                "predicted_domain": None,
                "error": None,
                "timing": {},
            }
            results.append(result)
            counters["ok"] += 1
            print(
                f"[{index}/{total}] {row_id}  planned  expected={expected or '?'}",
                file=sys.stderr,
            )
            continue

        try:
            state = run_pipeline(str(row["query"]).strip())
            result = _build_row_result(row, state)
        except Exception as exc:
            result = _build_row_result(row, None, error=str(exc))
            print(
                f"  ERROR on {row_id}: {exc}",
                file=sys.stderr,
            )
            # Keep traceback for debugging without aborting the run.
            print(traceback.format_exc(), file=sys.stderr)

        results.append(result)
        _print_progress(index, total, result)

        status = result.get("status")
        if status == "ERROR":
            counters["errors"] += 1
        elif status == "PASS":
            counters["pass"] += 1
            counters["ok"] += 1
        elif status == "FAIL":
            counters["fail"] += 1
            counters["ok"] += 1
        else:
            counters["unknown"] += 1
            counters["ok"] += 1

    return results, counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run data/eval_queries.jsonl through ARCS and save Phase 2 "
            "experiment artifacts. Does not write request logs or apply feedback."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Eval JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run at most N rows after filters",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated eval ids to include (e.g. eval-001,eval-002)",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated expected domains (e.g. CODING,MEDICAL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and counts; do not call the pipeline or write artifacts",
    )
    parser.add_argument(
        "--name",
        default="pipeline-eval",
        help="Experiment name (default: pipeline-eval)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress pipeline progress (eval progress still prints)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full experiment dict to stdout",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write experiment artifacts",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: eval file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None and args.limit < 0:
        print("Error: --limit must be >= 0", file=sys.stderr)
        sys.exit(1)

    try:
        id_filter = _parse_csv_set(args.ids, label="--ids")
        domain_filter = _parse_csv_set(args.domains, label="--domains")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if domain_filter is not None:
        normalized_domains: set[str] = set()
        for raw in domain_filter:
            domain = _normalize_domain(raw)
            if domain is None:
                print(
                    f"Error: invalid domain in --domains: {raw!r} "
                    f"(choose from {', '.join(VALID_DOMAINS)})",
                    file=sys.stderr,
                )
                sys.exit(1)
            normalized_domains.add(domain)
        domain_filter = normalized_domains

    rows = _load_rows(args.input)
    rows = _filter_rows(
        rows,
        ids=id_filter,
        domains=domain_filter,
        limit=args.limit,
    )
    if not rows:
        print("Error: no eval rows matched filters", file=sys.stderr)
        sys.exit(1)

    progress.set_verbose(not args.quiet and not args.dry_run)

    planned_counts = Counter(
        _normalize_domain(row.get("expected_domain")) or "UNKNOWN" for row in rows
    )
    print(
        f"Loaded {len(rows)} eval row(s) from {args.input}"
        + (" [dry-run]" if args.dry_run else "")
        + (" [no-save]" if args.no_save else ""),
        file=sys.stderr,
    )
    for domain in VALID_DOMAINS:
        print(f"  {domain}: {planned_counts.get(domain, 0)}", file=sys.stderr)

    results, counters = run_eval(rows, dry_run=args.dry_run)

    if args.dry_run:
        experiment = aggregate_experiment(
            args.name,
            router=None,
            pipeline=None,
            meta={
                "dry_run": True,
                "input": str(args.input),
                "n_planned": len(rows),
                "planned_domain_counts": {
                    domain: planned_counts.get(domain, 0) for domain in VALID_DOMAINS
                },
                "counters": counters,
            },
        )
        _print_summary(experiment, dry_run=True, saved_to=None)
        if args.json:
            print(json.dumps(experiment, indent=2, default=str))
        return

    router = router_accuracy(results)
    pipeline = pipeline_summary(results)
    experiment = aggregate_experiment(
        args.name,
        router=router,
        pipeline=pipeline,
        meta={
            "input": str(args.input),
            "n_rows": len(results),
            "counters": counters,
            "rows": results,
        },
    )

    saved_to: Path | None = None
    if not args.no_save:
        saved_to = save_experiment(
            experiment,
            name=args.name,
            output_dir=args.output_dir,
        )
        # Reload so meta includes run_id / git_commit written by save_experiment.
        from arcs.eval.experiments import load_experiment

        experiment = load_experiment(saved_to)

    _print_summary(experiment, dry_run=False, saved_to=saved_to)
    if args.json:
        print(json.dumps(experiment, indent=2, default=str))


if __name__ == "__main__":
    main()
