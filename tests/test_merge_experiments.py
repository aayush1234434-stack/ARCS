"""Unit tests for scripts/merge_experiments.merge_experiment_rows — no I/O."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "merge_experiments", _ROOT / "scripts" / "merge_experiments.py"
)
merge_experiments = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_experiments)

merge_experiment_rows = merge_experiments.merge_experiment_rows


def _row(rid: str, status: str, domain: str = "CODING", predicted: str | None = None) -> dict:
    return {
        "id": rid,
        "query": f"q-{rid}",
        "expected_domain": domain,
        "predicted_domain": predicted or domain,
        "status": status,
        "verification": {"verdict": status if status in ("PASS", "FAIL") else None},
        "timing": {"total_ms": 100},
    }


def _exp(created_at: str, rows: list[dict]) -> dict:
    return {"name": "e", "meta": {"created_at": created_at, "rows": rows}}


def test_disjoint_rows_are_unioned():
    a = _exp("2026-01-01", [_row("eval-001", "PASS")])
    b = _exp("2026-01-02", [_row("eval-002", "FAIL")])
    merged = merge_experiment_rows([a, b])
    ids = {r["id"] for r in merged}
    assert ids == {"eval-001", "eval-002"}


def test_non_error_beats_error_regardless_of_recency():
    # Older run has PASS, newer run has ERROR for the same id -> keep PASS.
    older = _exp("2026-01-01", [_row("eval-001", "PASS")])
    newer = _exp("2026-01-02", [_row("eval-001", "ERROR")])
    merged = merge_experiment_rows([older, newer])
    assert len(merged) == 1
    assert merged[0]["status"] == "PASS"


def test_error_is_replaced_by_later_success():
    older = _exp("2026-01-01", [_row("eval-001", "ERROR")])
    newer = _exp("2026-01-02", [_row("eval-001", "PASS")])
    merged = merge_experiment_rows([older, newer])
    assert merged[0]["status"] == "PASS"


def test_newer_completed_wins_between_two_completed():
    older = _exp("2026-01-01", [_row("eval-001", "FAIL")])
    newer = _exp("2026-01-02", [_row("eval-001", "PASS")])
    merged = merge_experiment_rows([older, newer])
    assert merged[0]["status"] == "PASS"


def test_all_error_keeps_newer_error():
    older = _exp("2026-01-01", [_row("eval-001", "ERROR", predicted=None)])
    newer_row = _row("eval-001", "ERROR")
    newer_row["error"] = "newer error"
    newer = _exp("2026-01-02", [newer_row])
    merged = merge_experiment_rows([older, newer])
    assert merged[0].get("error") == "newer error"


def test_preserves_first_seen_order():
    a = _exp("2026-01-01", [_row("eval-003", "PASS"), _row("eval-001", "PASS")])
    b = _exp("2026-01-02", [_row("eval-002", "FAIL")])
    merged = merge_experiment_rows([a, b])
    assert [r["id"] for r in merged] == ["eval-003", "eval-001", "eval-002"]
