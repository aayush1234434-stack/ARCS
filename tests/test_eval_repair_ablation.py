"""Tests for RQ1-bis repair ablation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_repair_ablation import (
    _build_table,
    _eval_queries_acc,
    _filter_rows,
    _pick_winner,
    _resolve_arm_models,
)


def test_resolve_arm_models_bootstrap() -> None:
    arms = _resolve_arm_models(corpus="bootstrap")
    assert set(arms) == {"arm0", "arm1", "arm2"}
    assert "rq1-run-a" in str(arms["arm1"]["model_dir"])
    assert "rq1-run-b" in str(arms["arm2"]["model_dir"])


def test_resolve_arm_models_real() -> None:
    arms = _resolve_arm_models(corpus="real")
    assert "rq1-v2-run-a" in str(arms["arm1"]["model_dir"])
    assert "rq1-v2-run-b" in str(arms["arm2"]["model_dir"])


def test_build_table_and_winner() -> None:
    arms = {
        "arm0": {
            "label": "pre",
            "model_dir": Path("/pre"),
            "eval_result": {"metrics": {"accuracy": 0.90, "n": 10}},
        },
        "arm1": {
            "label": "run a",
            "model_dir": Path("/a"),
            "eval_result": {"metrics": {"accuracy": 0.95, "n": 10}},
        },
        "arm2": {
            "label": "run b",
            "model_dir": Path("/b"),
            "eval_result": {"metrics": {"accuracy": 0.93, "n": 10}},
        },
    }
    table = _build_table(arms)
    assert len(table) == 3
    assert table[0]["delta_vs_arm0"] == 0.0
    assert table[1]["delta_vs_arm0"] == pytest.approx(0.05)
    assert table[2]["delta_vs_arm0"] == pytest.approx(0.03)
    assert _pick_winner(table) == "arm1"


def test_pick_winner_tie() -> None:
    table = [
        {"arm": "arm0", "router_eval_acc": 0.9, "delta_vs_arm0": 0.0},
        {"arm": "arm1", "router_eval_acc": 0.95, "delta_vs_arm0": 0.05},
        {"arm": "arm2", "router_eval_acc": 0.95, "delta_vs_arm0": 0.05},
    ]
    assert _pick_winner(table) == "tie"


def test_filter_rows_domain_and_limit() -> None:
    rows = [
        {"id": "eval-001", "query": "a", "expected_domain": "CODING"},
        {"id": "eval-002", "query": "b", "expected_domain": "MEDICAL"},
        {"id": "eval-003", "query": "c", "expected_domain": "CODING"},
    ]
    out = _filter_rows(rows, ids=None, domains={"CODING"}, limit=1)
    assert len(out) == 1
    assert out[0]["id"] == "eval-001"


def test_eval_queries_acc() -> None:
    assert _eval_queries_acc({"metrics": {"accuracy": 0.9375}}) == 0.9375
    assert _eval_queries_acc({}) is None
