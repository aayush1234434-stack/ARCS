#!/usr/bin/env python3
"""Evaluate routed ARCS specialists with runtime verification disabled.

Each query is routed and answered by exactly one specialist call. No generated
tests, sandbox execution, judge gate, or retry can influence the delivered
answer. A specification and independent judge are invoked only afterward to
produce the same proxy outcome metric used by the other benchmark conditions;
their latency and usage are stored separately as ``evaluation`` overhead.
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

from arcs import config, progress, router
from arcs.clients.rate_limit import is_groq_tpd_exhausted
from arcs.clients.usage import combine_usage
from arcs.eval.experiments import load_experiment, save_experiment
from arcs.eval.metrics import VALID_DOMAINS, aggregate_experiment, pipeline_summary, router_accuracy
from arcs.pipelines import resolve_pipeline
from arcs.verification import judge, spec_generator

DEFAULT_INPUT = config.DATA_DIR / "eval_queries.jsonl"
DOMAIN_SET = frozenset(VALID_DOMAINS)
KIND = "arcs_no_verification"
EXIT_TPD_EXHAUSTED = 2


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not str(row.get("query") or "").strip():
                raise ValueError(f"invalid benchmark row on line {line_number}")
            rows.append(row)
    return rows


def _normalize_domain(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    domain = value.strip().upper()
    return domain if domain in DOMAIN_SET else None


def _answer_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("answer", "explanation", "complexity", "edge_cases", "caveats"):
        value = result.get(key)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return "\n\n".join(parts)


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _run_one(row: dict[str, Any]) -> dict[str, Any]:
    query = str(row["query"]).strip()
    expected = _normalize_domain(row.get("expected_domain"))
    result: dict[str, Any] = {
        "id": row.get("id"),
        "query": query,
        "expected_domain": expected,
        "runtime_verification_enabled": False,
        "error": None,
    }
    timing: dict[str, int] = {}
    started = time.perf_counter()

    route_started = time.perf_counter()
    route = router.route(query)
    timing["route_ms"] = _elapsed_ms(route_started)
    pipeline = resolve_pipeline(
        str(route.get("domain") or "GENERAL"),
        use_fallback=bool(route.get("use_fallback")),
    )
    model = pipeline.resolve_model()

    generation_started = time.perf_counter()
    specialist = pipeline.specialist.run(query, model=model)
    timing["specialist_ms"] = _elapsed_ms(generation_started)
    timing["runtime_total_ms"] = _elapsed_ms(started)
    answer = _answer_text(specialist)

    result.update(
        {
            "predicted_domain": route.get("domain"),
            "router_confidence": route.get("confidence"),
            "use_fallback": bool(route.get("use_fallback")),
            "pipeline_id": pipeline.pipeline_id,
            "verifier": "posthoc_llm_judge",
            "specialist": {**specialist, "answer": answer},
            "answer": answer,
        }
    )

    if not answer:
        result["status"] = "FAIL"
        result["verification"] = {
            "verdict": "FAIL",
            "score": 0.0,
            "explanation": "specialist returned an empty answer",
        }
        result["usage"] = {
            "total": specialist.get("usage", {}),
            "components": {"generation": specialist.get("usage", {})},
            "evaluation": {},
        }
        timing["total_ms"] = _elapsed_ms(started)
        result["timing"] = timing
        return result

    evaluation_started = time.perf_counter()
    specification = spec_generator.run(query)
    verification = judge.run(
        question=query,
        answer=answer,
        specification=specification,
    )
    timing["evaluation_ms"] = _elapsed_ms(evaluation_started)
    timing["verification_ms"] = timing["evaluation_ms"]
    timing["total_ms"] = _elapsed_ms(started)

    verdict = str(verification.get("verdict") or "").upper()
    result["status"] = verdict if verdict in {"PASS", "FAIL"} else "UNKNOWN"
    result["specification"] = specification
    result["verification"] = verification
    result["usage"] = {
        "total": specialist.get("usage", {}),
        "components": {"generation": specialist.get("usage", {})},
        "evaluation": {
            "total": combine_usage(
                (specification.get("usage", {}), verification.get("usage", {}))
            ),
            "components": {
                "specification": specification.get("usage", {}),
                "judge": verification.get("usage", {}),
            },
        },
        "cost_usd": None,
        "cost_note": "Runtime usage excludes the post-hoc benchmark evaluator.",
    }
    result["timing"] = timing
    return result


def _error_row(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "query": str(row.get("query") or "").strip(),
        "expected_domain": _normalize_domain(row.get("expected_domain")),
        "predicted_domain": None,
        "runtime_verification_enabled": False,
        "status": "ERROR",
        "error": str(exc),
        "error_class": type(exc).__name__,
        "verification": {"verdict": None, "score": None},
        "timing": {},
        "usage": {},
    }


def run_eval(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
    sleep_between: float = 0.0,
) -> tuple[list[dict[str, Any]], bool]:
    results: list[dict[str, Any]] = []
    tpd_exhausted = False
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id") or f"row-{index}"
        if dry_run:
            result = {
                "id": row.get("id"),
                "query": str(row.get("query") or "").strip(),
                "expected_domain": _normalize_domain(row.get("expected_domain")),
                "status": "PLANNED",
            }
        else:
            try:
                result = _run_one(row)
            except Exception as exc:  # noqa: BLE001 - preserve partial evidence
                result = _error_row(row, exc)
                print(f"  ERROR on {row_id}: {exc}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                if is_groq_tpd_exhausted(exc):
                    tpd_exhausted = True
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {row_id} status={result.get('status')} "
            f"routed={result.get('predicted_domain') or '?'}",
            file=sys.stderr,
        )
        if tpd_exhausted:
            break
        if sleep_between > 0 and index < len(rows):
            time.sleep(sleep_between)
    return results, tpd_exhausted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--name", default="arcs-no-verification")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.sleep_between < 0:
        parser.error("--sleep-between must be >= 0")
    progress.set_verbose(not args.quiet)

    rows = _load_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("no benchmark rows selected")
    counts = Counter(_normalize_domain(row.get("expected_domain")) for row in rows)
    print(f"Loaded {len(rows)} row(s) from {args.input} [ARCS no verification]", file=sys.stderr)
    for domain in VALID_DOMAINS:
        print(f"  {domain}: {counts.get(domain, 0)}", file=sys.stderr)

    results, tpd_exhausted = run_eval(
        rows,
        dry_run=args.dry_run,
        sleep_between=args.sleep_between,
    )
    if args.dry_run:
        return

    experiment = aggregate_experiment(
        args.name,
        router=router_accuracy(results),
        pipeline=pipeline_summary(results),
        meta={
            "input": str(args.input),
            "n_rows": len(results),
            "condition": "routed specialist; runtime verification disabled; post-hoc judge only",
            "rows": results,
        },
    )
    experiment["kind"] = KIND
    saved_to: Path | None = None
    if not args.no_save:
        saved_to = save_experiment(experiment, name=args.name, output_dir=args.output_dir)
        experiment = load_experiment(saved_to)
    summary = experiment.get("pipeline") or {}
    print(
        f"PASS={summary.get('status_counts', {}).get('PASS', 0)} "
        f"FAIL={summary.get('status_counts', {}).get('FAIL', 0)} "
        f"ERROR={summary.get('status_counts', {}).get('ERROR', 0)}",
        file=sys.stderr,
    )
    if saved_to:
        print(f"saved: {saved_to}", file=sys.stderr)
    if args.json:
        print(json.dumps(experiment, indent=2, default=str))
    if tpd_exhausted:
        raise SystemExit(EXIT_TPD_EXHAUSTED)


if __name__ == "__main__":
    main()
