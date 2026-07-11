#!/usr/bin/env python3
"""End-to-end pipeline smoke test on five fixed held-out eval queries.

One query per domain plus one coding-in-prose tricky row. Asserts each run
completes with status PASS or FAIL (zero ERROR / UNKNOWN rows).

Requires GROQ_API_KEY, NVIDIA_API_KEY, and a trained router under
``artifacts/router-model/``. Does not write to ``logs/requests.jsonl``.

Exit codes:
    0  — all 5 queries completed (PASS or FAIL); zero ERROR
    1  — one or more ERROR / UNKNOWN rows (see ``error_class`` in JSON output)
    2  — setup failure (missing keys, router checkpoint, eval file)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config, progress
from arcs.main import PipelineError, classify_pipeline_error, run_pipeline

EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"

# Fixed smoke set: 1 per domain + coding-in-prose (eval-042).
SMOKE_IDS: tuple[str, ...] = (
    "eval-024",  # LEGAL
    "eval-013",  # MEDICAL
    "eval-033",  # GENERAL
    "eval-001",  # CODING
    "eval-042",  # CODING (prose / tricky)
)

EXPECTED_DOMAINS: dict[str, str] = {
    "eval-024": "LEGAL",
    "eval-013": "MEDICAL",
    "eval-033": "GENERAL",
    "eval-001": "CODING",
    "eval-042": "CODING",
}

COMPLETED_STATUSES = frozenset({"PASS", "FAIL"})

EXIT_OK = 0
EXIT_SMOKE_FAIL = 1
EXIT_SETUP = 2


@dataclass(frozen=True)
class SmokeCase:
    id: str
    query: str
    expected_domain: str
    notes: str | None = None


def _load_eval_index(path: Path = EVAL_QUERIES) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Eval queries not found: {path}")

    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row on line {line_number} of {path}")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"Missing id on line {line_number} of {path}")
            index[row_id] = row
    return index


def load_smoke_cases() -> list[SmokeCase]:
    index = _load_eval_index()
    cases: list[SmokeCase] = []
    for row_id in SMOKE_IDS:
        if row_id not in index:
            raise KeyError(f"Smoke id {row_id!r} not found in {EVAL_QUERIES}")
        row = index[row_id]
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Smoke id {row_id!r} has empty query")
        expected = EXPECTED_DOMAINS[row_id]
        file_expected = str(row.get("expected_domain") or "").strip().upper()
        if file_expected and file_expected != expected:
            raise ValueError(
                f"Smoke id {row_id!r}: expected {expected}, "
                f"eval file has {file_expected}"
            )
        notes = row.get("notes")
        cases.append(
            SmokeCase(
                id=row_id,
                query=query.strip(),
                expected_domain=expected,
                notes=str(notes) if notes is not None else None,
            )
        )
    return cases


def _derive_status(state: dict[str, Any] | None, *, error: str | None) -> str:
    if error:
        return "ERROR"
    if not isinstance(state, dict):
        return "UNKNOWN"
    if state.get("error"):
        return "ERROR"
    verdict = (state.get("verification") or {}).get("verdict")
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def _check_api_keys() -> list[str]:
    missing: list[str] = []
    if not os.getenv("GROQ_API_KEY", "").strip():
        missing.append("GROQ_API_KEY")
    if not os.getenv("NVIDIA_API_KEY", "").strip():
        missing.append("NVIDIA_API_KEY")
    return missing


def _run_case(case: SmokeCase) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": case.id,
        "expected_domain": case.expected_domain,
        "query": case.query,
        "notes": case.notes,
    }
    try:
        state = run_pipeline(case.query)
    except PipelineError as exc:
        state = exc.state
        result["status"] = _derive_status(state, error=str(state.get("error") or exc))
        result["error"] = str(state.get("error") or exc)
        result["error_class"] = str(state.get("error_class") or classify_pipeline_error(exc))
        return result
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        result["error_class"] = classify_pipeline_error(exc)
        result["traceback"] = traceback.format_exc()
        return result

    route = state.get("route") or {}
    verification = state.get("verification") or {}
    timing = state.get("timing") or {}

    result["status"] = _derive_status(state, error=None)
    result["predicted_domain"] = route.get("domain")
    result["router_confidence"] = route.get("confidence")
    result["verdict"] = verification.get("verdict")
    result["score"] = verification.get("score")
    result["timing_ms"] = timing.get("total_ms") if isinstance(timing, dict) else None
    if result["status"] == "ERROR":
        result["error"] = str(state.get("error") or "pipeline error")
        result["error_class"] = str(state.get("error_class") or "unknown")
    elif result["status"] == "UNKNOWN":
        result["error_class"] = "unknown"
        result["error"] = "pipeline finished without PASS/FAIL verdict"
    return result


def _summarize_results(results: list[dict[str, Any]], *, total: int) -> dict[str, Any]:
    pass_count = sum(1 for row in results if row.get("status") == "PASS")
    fail_count = sum(1 for row in results if row.get("status") == "FAIL")
    error_rows = [row for row in results if row.get("status") not in COMPLETED_STATUSES]
    errors = len(error_rows)
    error_classes = Counter(
        str(row.get("error_class") or "unknown") for row in error_rows
    )
    complete = pass_count + fail_count
    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "complete": complete,
        "errors": errors,
        "error_classes": dict(sorted(error_classes.items())),
        "success": errors == 0 and complete == total,
    }


def run_smoke(*, dry_run: bool = False) -> dict[str, Any]:
    cases = load_smoke_cases()
    if dry_run:
        return {
            "dry_run": True,
            "cases": [
                {
                    "id": c.id,
                    "expected_domain": c.expected_domain,
                    "query": c.query,
                    "notes": c.notes,
                }
                for c in cases
            ],
            "summary": {
                "total": len(cases),
                "complete": 0,
                "errors": 0,
                "planned": len(cases),
            },
        }

    missing_keys = _check_api_keys()
    if missing_keys:
        raise RuntimeError(
            "Missing API keys: "
            + ", ".join(missing_keys)
            + " (required for live smoke e2e)"
        )

    router_config = config.ROUTER_MODEL_DIR / "config.json"
    if not router_config.is_file():
        raise FileNotFoundError(
            f"Router checkpoint not found: {router_config}. "
            "Train with: python -m arcs.router.train"
        )

    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case.id} ({case.expected_domain})",
            file=sys.stderr,
        )
        row = _run_case(case)
        results.append(row)
        status = row.get("status")
        if status in COMPLETED_STATUSES:
            print(
                f"  OK status={status} routed={row.get('predicted_domain')} "
                f"verdict={row.get('verdict')}",
                file=sys.stderr,
            )
        else:
            error_class = row.get("error_class") or "unknown"
            print(
                f"  ERROR status={status} error_class={error_class} "
                f"error={row.get('error', '')}",
                file=sys.stderr,
            )

    summary = _summarize_results(results, total=len(cases))
    return {
        "dry_run": False,
        "results": results,
        "summary": summary,
    }


def _exit_code_for_report(report: dict[str, Any], *, dry_run: bool) -> int:
    if dry_run:
        return EXIT_OK
    summary = report.get("summary") or {}
    if summary.get("errors", 0) == 0 and summary.get("complete") == summary.get("total"):
        return EXIT_OK
    return EXIT_SMOKE_FAIL


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run five fixed eval queries through the full ARCS pipeline. "
            f"Exit {EXIT_OK} only when all complete with PASS or FAIL (zero ERROR)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned queries only; no API calls",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print report as JSON to stdout",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress pipeline progress logs",
    )
    args = parser.parse_args()

    progress.set_verbose(not args.quiet and not args.json)

    try:
        report = run_smoke(dry_run=args.dry_run)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(EXIT_SETUP)

    if args.json:
        payload = dict(report)
        payload["exit_code"] = _exit_code_for_report(report, dry_run=args.dry_run)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.dry_run:
        for case in report.get("cases", []):
            print(
                f"{case['id']}  {case['expected_domain']:8s}  {case['query'][:72]}",
                file=sys.stderr,
            )
        print(f"\n(dry-run: {report['summary']['planned']} queries planned)", file=sys.stderr)
    else:
        summary = report["summary"]
        print(
            f"\nSmoke e2e: {summary['complete']}/{summary['total']} complete "
            f"(PASS={summary['pass']} FAIL={summary['fail']}), "
            f"errors={summary['errors']}",
            file=sys.stderr,
        )
        if summary["errors"]:
            print("ERROR breakdown by error_class:", file=sys.stderr)
            for bucket, count in sorted((summary.get("error_classes") or {}).items()):
                print(f"  {bucket}: {count}", file=sys.stderr)
        print(
            f"exit_code={_exit_code_for_report(report, dry_run=False)} "
            f"(0=ok, {EXIT_SMOKE_FAIL}=smoke fail, {EXIT_SETUP}=setup)",
            file=sys.stderr,
        )

    sys.exit(_exit_code_for_report(report, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
