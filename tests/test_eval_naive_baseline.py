"""Tests for the naive single-LLM baseline eval harness."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.eval.compare import pass_stats
from arcs.eval.metrics import aggregate_experiment, pipeline_summary, router_accuracy
from scripts.eval_naive_baseline import (
    KIND,
    _filter_rows,
    _run_one,
    _verdict_to_status,
)


def _make_experiment(name: str, rows: list[dict]) -> dict:
    exp = aggregate_experiment(
        name,
        router=router_accuracy(rows),
        pipeline=pipeline_summary(rows),
        meta={"rows": rows},
    )
    exp["kind"] = KIND
    return exp


def test_verdict_to_status() -> None:
    assert _verdict_to_status({"verdict": "PASS"}) == "PASS"
    assert _verdict_to_status({"verdict": "fail"}) == "FAIL"
    assert _verdict_to_status({}) == "UNKNOWN"


def test_pass_stats_ignores_errors() -> None:
    rows = [
        {"status": "PASS", "expected_domain": "CODING", "verification": {"verdict": "PASS"}},
        {"status": "PASS", "expected_domain": "MEDICAL", "verification": {"verdict": "PASS"}},
        {"status": "FAIL", "expected_domain": "LEGAL", "verification": {"verdict": "FAIL"}},
        {"status": "ERROR", "expected_domain": "CODING", "error": "boom",
         "verification": {"verdict": None}},
    ]
    exp = _make_experiment("naive-baseline-v1", rows)
    stats = pass_stats(exp)
    assert stats["pass"] == 2
    assert stats["fail"] == 1
    assert stats["error"] == 1
    assert stats["completed"] == 3
    # 2/3 completed rows passed → ~66.7%, errors excluded from the denominator.
    assert stats["pass_pct"] is not None
    assert abs(stats["pass_pct"] - (2 / 3 * 100)) < 1e-6


def test_pass_stats_no_completed_rows() -> None:
    rows = [{"status": "ERROR", "expected_domain": "CODING", "error": "x",
             "verification": {"verdict": None}}]
    stats = pass_stats(_make_experiment("naive", rows))
    assert stats["completed"] == 0
    assert stats["pass_pct"] is None


def test_filter_rows_by_domain_and_limit() -> None:
    rows = [
        {"id": "eval-001", "query": "a", "expected_domain": "CODING"},
        {"id": "eval-002", "query": "b", "expected_domain": "MEDICAL"},
        {"id": "eval-003", "query": "c", "expected_domain": "CODING"},
    ]
    coding = _filter_rows(rows, ids=None, domains={"CODING"}, limit=None)
    assert [r["id"] for r in coding] == ["eval-001", "eval-003"]

    limited = _filter_rows(rows, ids=None, domains=None, limit=2)
    assert len(limited) == 2


def test_run_one_uses_naive_answer_spec_and_judge(monkeypatch) -> None:
    """_run_one should skip routing and call the naive answer + shared spec/judge."""
    import scripts.eval_naive_baseline as mod

    monkeypatch.setattr(mod, "_naive_answer", lambda query, *, model: "42 is the answer")

    fake_spec = {
        "intent": "answer",
        "required_elements": ["a number"],
        "correctness_criteria": ["is numeric"],
        "disqualifying_conditions": [],
        "scope": "math",
    }

    import arcs.verification.spec_generator as spec_generator
    import arcs.verification.judge as judge

    monkeypatch.setattr(spec_generator, "run", lambda query: fake_spec)
    monkeypatch.setattr(
        judge,
        "run",
        lambda *, question, answer, specification: {
            "verdict": "PASS",
            "score": 0.9,
            "explanation": "ok",
        },
    )

    row = {"id": "eval-x", "query": "What is 6 times 7?", "expected_domain": "CODING"}
    result = _run_one(row, model="llama-3.3-70b-versatile")

    assert result["status"] == "PASS"
    assert result["predicted_domain"] is None  # no routing in the naive baseline
    assert result["pipeline_id"] == "NAIVE"
    assert result["verification"]["score"] == 0.9
    assert result["answer"] == "42 is the answer"


def test_compare_only_prints_delta(capsys) -> None:
    import scripts.eval_naive_baseline as mod

    naive = _make_experiment(
        "naive-baseline-v1",
        [
            {"status": "PASS", "expected_domain": "CODING", "verification": {"verdict": "PASS"}},
            {"status": "FAIL", "expected_domain": "MEDICAL", "verification": {"verdict": "FAIL"}},
        ],
    )
    arcs = _make_experiment(
        "post-fix-v2",
        [
            {"status": "PASS", "expected_domain": "CODING", "verification": {"verdict": "PASS"}},
            {"status": "PASS", "expected_domain": "MEDICAL", "verification": {"verdict": "PASS"}},
        ],
    )
    mod._print_comparison(naive, arcs)
    err = capsys.readouterr().err
    assert "Naive vs ARCS" in err
    assert "ARCS − naive:" in err
    assert "50.0%" in err
