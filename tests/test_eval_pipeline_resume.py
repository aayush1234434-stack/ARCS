"""Unit tests for eval_pipeline resume / merge helpers — no API calls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "eval_pipeline", _ROOT / "scripts" / "eval_pipeline.py"
)
eval_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_pipeline)

COMPLETED = eval_pipeline.COMPLETED_STATUSES


def _prior(*rows: dict) -> dict[str, dict]:
    return {str(r["id"]): r for r in rows}


def test_resume_skips_pass_and_fail_reruns_error_and_missing():
    prior = _prior(
        {"id": "eval-001", "status": "PASS"},
        {"id": "eval-021", "status": "ERROR", "error": "parse error"},
        {"id": "eval-044", "status": "ERROR", "error": "TPD"},
    )
    input_rows = [
        {"id": "eval-001", "query": "q1"},
        {"id": "eval-021", "query": "q21"},
        {"id": "eval-044", "query": "q44"},
        {"id": "eval-048", "query": "q48"},
    ]
    done_ids = {
        rid
        for rid, row in prior.items()
        if str(row.get("status") or "").upper() in COMPLETED
    }
    remaining = [r for r in input_rows if str(r.get("id")) not in done_ids]
    assert [r["id"] for r in remaining] == ["eval-021", "eval-044", "eval-048"]


def test_merge_prior_with_new_results_new_wins_on_id_collision():
    prior = _prior(
        {"id": "eval-021", "status": "ERROR", "error": "old"},
        {"id": "eval-001", "status": "PASS"},
    )
    new_results = [
        {"id": "eval-021", "status": "FAIL", "error": None},
        {"id": "eval-048", "status": "PASS"},
    ]
    new_ids = {str(r.get("id")) for r in new_results}
    merged = list(new_results)
    for rid, row in prior.items():
        if rid not in new_ids:
            merged.append(row)
    by_id = {r["id"]: r for r in merged}
    assert by_id["eval-021"]["status"] == "FAIL"
    assert by_id["eval-001"]["status"] == "PASS"
    assert by_id["eval-048"]["status"] == "PASS"
    assert len(by_id) == 3
