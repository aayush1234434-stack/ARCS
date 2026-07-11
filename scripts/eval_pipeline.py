"""CLI: run the held-out eval set through ARCS and save experiment artifacts.

Does not write to logs/requests.jsonl and does not collect feedback/attribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config, progress
from arcs.clients.rate_limit import is_groq_tpd_exhausted
from arcs.eval.experiments import load_experiment, save_experiment
from arcs.eval.metrics import (
    VALID_DOMAINS,
    aggregate_experiment,
    pipeline_summary,
    router_accuracy,
)
from arcs.main import PipelineError, classify_pipeline_error, run_pipeline

DEFAULT_INPUT = config.DATA_DIR / "eval_queries.jsonl"
DOMAIN_SET = frozenset(VALID_DOMAINS)

# Exit code used when the run stops early because Groq's daily token limit (TPD)
# is exhausted; partial results are saved and a resume command is printed.
EXIT_TPD_EXHAUSTED = 2

# Statuses considered "done" — resume will not re-run these.
COMPLETED_STATUSES = frozenset({"PASS", "FAIL"})


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
    error_class: str | None = None,
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
        result["error_class"] = (
            error_class
            or (state.get("error_class") if isinstance(state, dict) else None)
            or classify_pipeline_error(Exception(error))
        )
        if isinstance(state, dict):
            result["query_id"] = state.get("query_id")
            route = state.get("route") or {}
            pipeline = state.get("pipeline") or {}
            timing = state.get("timing") or {}
            predicted = route.get("domain") if isinstance(route, dict) else None
            confidence = route.get("confidence") if isinstance(route, dict) else None
            try:
                confidence_f = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_f = None
            result["predicted_domain"] = (
                str(predicted).strip().upper() if predicted is not None else None
            )
            result["router_confidence"] = confidence_f
            result["use_fallback"] = (
                bool(route.get("use_fallback")) if isinstance(route, dict) else None
            )
            result["pipeline_id"] = (
                pipeline.get("pipeline_id") if isinstance(pipeline, dict) else None
            )
            result["verifier"] = (
                pipeline.get("verifier") if isinstance(pipeline, dict) else None
            )
            result["timing"] = dict(timing) if isinstance(timing, dict) else {}
        else:
            result["query_id"] = None
            result["predicted_domain"] = None
            result["router_confidence"] = None
            result["use_fallback"] = None
            result["pipeline_id"] = None
            result["verifier"] = None
            result["timing"] = {}
        result["verification"] = {"verdict": None, "score": None}
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
    result["query_id"] = state.get("query_id")
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

    specification = state.get("specification")
    if isinstance(specification, dict) and specification:
        result["specification"] = specification

    specialist = state.get("specialist")
    tooling = state.get("tooling") or {}
    if isinstance(specialist, dict):
        specialist_out: dict[str, Any] = {}
        answer = specialist.get("answer")
        if isinstance(answer, str) and answer.strip():
            specialist_out["answer"] = answer
        test_cases = specialist.get("test_cases") or tooling.get("test_cases")
        if isinstance(test_cases, list) and test_cases:
            specialist_out["test_cases"] = test_cases
        pipeline_id = specialist.get("pipeline_id") or result.get("pipeline_id")
        if pipeline_id:
            specialist_out["pipeline_id"] = pipeline_id
        if specialist_out:
            result["specialist"] = specialist_out

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
    error_class = result.get("error_class")
    extra = f"  error={err}" if err else ""
    if error_class:
        extra += f"  error_class={error_class}"
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
    error_rows = [
        row for row in (meta.get("rows") or []) if row.get("status") == "ERROR"
    ]
    if error_rows:
        breakdown = Counter(str(row.get("error_class") or "unknown") for row in error_rows)
        print("ERROR breakdown by error_class:", file=sys.stderr)
        for cls, count in sorted(breakdown.items()):
            print(f"  {cls}: {count}", file=sys.stderr)
    latency = (pipeline.get("latency_ms") or {}).get("total_ms") or {}
    if latency.get("count"):
        print(
            f"latency:  mean={latency.get('mean')}ms  "
            f"p50={latency.get('p50')}ms  p95={latency.get('p95')}ms",
            file=sys.stderr,
        )
    if saved_to is not None:
        print(f"saved:    {saved_to}", file=sys.stderr)


def _tally(counters: dict[str, int], status: str | None) -> None:
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


def run_eval(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
    sleep_between: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    """Run eval rows.

    Returns ``(per-row results, counters, tpd_exhausted)``. When the Groq daily
    token limit is hit, the loop stops early, records the offending row as ERROR,
    and sets ``tpd_exhausted=True`` so the caller can save partial results.
    """
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
    tpd_exhausted = False

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
        except PipelineError as exc:
            state = exc.state
            result = _build_row_result(
                row,
                state,
                error=str(state.get("error") or exc),
                error_class=str(state.get("error_class") or "unknown"),
            )
            print(
                f"  ERROR on {row_id} (query_id={state.get('query_id')}): "
                f"{state.get('error')} [{state.get('error_class')}]",
                file=sys.stderr,
            )
            if is_groq_tpd_exhausted(exc.__cause__ or exc):
                results.append(result)
                _print_progress(index, total, result)
                counters["errors"] += 1
                tpd_exhausted = True
                print(
                    "\nGroq daily token limit (TPD) exhausted — stopping early.",
                    file=sys.stderr,
                )
                break
        except Exception as exc:
            result = _build_row_result(
                row,
                None,
                error=str(exc),
                error_class=classify_pipeline_error(exc),
            )
            print(f"  ERROR on {row_id}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if is_groq_tpd_exhausted(exc.__cause__ or exc):
                results.append(result)
                _print_progress(index, total, result)
                counters["errors"] += 1
                tpd_exhausted = True
                print(
                    "\nGroq daily token limit (TPD) exhausted — stopping early.",
                    file=sys.stderr,
                )
                break

        results.append(result)
        _print_progress(index, total, result)
        _tally(counters, result.get("status"))

        if sleep_between > 0 and index < total:
            time.sleep(sleep_between)

    return results, counters, tpd_exhausted


def _load_prior_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Return {id: row} from a prior experiment's meta.rows."""
    experiment = load_experiment(path)
    prior = (experiment.get("meta") or {}).get("rows") or []
    by_id: dict[str, dict[str, Any]] = {}
    for row in prior:
        if isinstance(row, dict) and row.get("id"):
            by_id[str(row["id"])] = row
    return by_id


def _finalize_and_save(
    results: list[dict[str, Any]],
    *,
    name: str,
    input_path: Path,
    counters: dict[str, int],
    output_dir: Path | None,
    no_save: bool,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Recompute metrics over results and (optionally) save the experiment."""
    router = router_accuracy(results)
    pipeline = pipeline_summary(results)
    meta: dict[str, Any] = {
        "input": str(input_path),
        "n_rows": len(results),
        "counters": counters,
        "rows": results,
    }
    if extra_meta:
        meta.update(extra_meta)
    experiment = aggregate_experiment(name, router=router, pipeline=pipeline, meta=meta)

    saved_to: Path | None = None
    if not no_save:
        saved_to = save_experiment(experiment, name=name, output_dir=output_dir)
        experiment = load_experiment(saved_to)
    return experiment, saved_to


def _print_resume_hint(
    results: list[dict[str, Any]],
    *,
    saved_to: Path | None,
    args: argparse.Namespace,
) -> None:
    """Print a ready-to-paste command to resume after TPD exhaustion."""
    incomplete = [
        str(r.get("id"))
        for r in results
        if r.get("id")
        and str(r.get("status") or "").upper() not in COMPLETED_STATUSES
    ]
    print("\n=== Resume after quota resets ===", file=sys.stderr)
    if saved_to is None:
        print(
            "Partial results were NOT saved (--no-save); re-run without --no-save "
            "to enable resume.",
            file=sys.stderr,
        )
        return
    cmd = (
        f"python scripts/eval_pipeline.py --name {args.name} "
        f"--resume-from {saved_to} --sleep-between {max(args.sleep_between, 1.0):g}"
    )
    print(f"Partial experiment saved to: {saved_to}", file=sys.stderr)
    if incomplete:
        print(f"Incomplete ids ({len(incomplete)}): {','.join(incomplete)}", file=sys.stderr)
    print("Resume with:", file=sys.stderr)
    print(f"  {cmd}", file=sys.stderr)


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
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Prior experiment dir or experiment.json. Rows already PASS/FAIL "
            "there are skipped; only ERROR/missing ids are re-run, and results "
            "are merged into the saved experiment."
        ),
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        metavar="N",
        help="Seconds to sleep between rows to ease rate limits (default: 0)",
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

    if args.sleep_between < 0:
        print("Error: --sleep-between must be >= 0", file=sys.stderr)
        sys.exit(1)

    prior_rows: dict[str, dict[str, Any]] = {}
    if args.resume_from is not None:
        try:
            prior_rows = _load_prior_rows(args.resume_from)
        except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
            print(f"Error: could not load --resume-from: {exc}", file=sys.stderr)
            sys.exit(1)

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

    # Resume: skip rows already completed (PASS/FAIL) in the prior experiment.
    if prior_rows and not args.dry_run:
        done_ids = {
            rid
            for rid, row in prior_rows.items()
            if str(row.get("status") or "").upper() in COMPLETED_STATUSES
        }
        remaining = [r for r in rows if str(r.get("id")) not in done_ids]
        print(
            f"Resume: {len(done_ids)} row(s) already complete, "
            f"{len(remaining)} to (re)run.",
            file=sys.stderr,
        )
        rows = remaining
        if not rows:
            print("Nothing to resume — all rows already complete.", file=sys.stderr)
            # Still rebuild/save the merged experiment from prior rows.
            merged = list(prior_rows.values())
            experiment, saved_to = _finalize_and_save(
                merged,
                name=args.name,
                input_path=args.input,
                counters={"total": len(merged), "resumed": len(merged)},
                output_dir=args.output_dir,
                no_save=args.no_save,
                extra_meta={"resumed_from": str(args.resume_from)},
            )
            _print_summary(experiment, dry_run=False, saved_to=saved_to)
            if args.json:
                print(json.dumps(experiment, indent=2, default=str))
            return

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

    results, counters, tpd_exhausted = run_eval(
        rows,
        dry_run=args.dry_run,
        sleep_between=args.sleep_between,
    )

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

    # Merge prior completed rows (resume) with the freshly-run rows. New results
    # win on id collision so a re-run ERROR->PASS is reflected.
    if prior_rows:
        new_ids = {str(r.get("id")) for r in results}
        merged = list(results)
        for rid, row in prior_rows.items():
            if rid not in new_ids:
                merged.append(row)
        results = merged
        counters["resumed_total"] = len(results)

    extra_meta: dict[str, Any] = {}
    if args.resume_from is not None:
        extra_meta["resumed_from"] = str(args.resume_from)

    experiment, saved_to = _finalize_and_save(
        results,
        name=args.name,
        input_path=args.input,
        counters=counters,
        output_dir=args.output_dir,
        no_save=args.no_save,
        extra_meta=extra_meta or None,
    )

    _print_summary(experiment, dry_run=False, saved_to=saved_to)
    if args.json:
        print(json.dumps(experiment, indent=2, default=str))

    if tpd_exhausted:
        _print_resume_hint(results, saved_to=saved_to, args=args)
        sys.exit(EXIT_TPD_EXHAUSTED)


if __name__ == "__main__":
    main()
