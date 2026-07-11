"""Unit tests for arcs.eval.compare — fixture-only, no API calls."""

from __future__ import annotations

import pytest

from arcs.eval.compare import diff_experiments, format_diff


def _router_fixture(*, accuracy: float, macro_f1: float, coding_f1: float) -> dict:
    return {
        "name": "router-fixture",
        "kind": "router",
        "meta": {"run_id": "fixture-router"},
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": accuracy,
            "per_class": {
                "CODING": {"precision": 1.0, "recall": 1.0, "f1": coding_f1, "support": 10},
                "MEDICAL": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10},
                "LEGAL": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10},
                "GENERAL": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10},
            },
        },
        "router": {"accuracy": accuracy, "macro_f1": macro_f1},
        "pipeline": None,
    }


def _pipeline_fixture(
    *,
    pass_rate: float,
    router_acc: float,
    mean_ms: float,
    p95_ms: float,
    coding_pass: float,
) -> dict:
    return {
        "name": "pipeline-fixture",
        "kind": "pipeline",
        "meta": {"run_id": "fixture-pipeline"},
        "router": {"accuracy": router_acc, "n": 4},
        "pipeline": {
            "n": 4,
            "status_rates": {
                "PASS": pass_rate,
                "FAIL": 1.0 - pass_rate,
                "UNKNOWN": 0.0,
                "ERROR": 0.0,
            },
            "latency_ms": {
                "total_ms": {"count": 4, "mean": mean_ms, "p50": mean_ms, "p95": p95_ms}
            },
            "per_domain": {
                "CODING": {"n": 1, "pass_rate": coding_pass},
                "MEDICAL": {"n": 1, "pass_rate": 1.0},
                "LEGAL": {"n": 1, "pass_rate": 0.0},
                "GENERAL": {"n": 1, "pass_rate": 1.0},
            },
        },
    }


def test_diff_router_improvements():
    a = _router_fixture(accuracy=0.90, macro_f1=0.88, coding_f1=0.85)
    b = _router_fixture(accuracy=0.95, macro_f1=0.94, coding_f1=0.99)
    diff = diff_experiments(a, b)
    assert diff["kind"] == "router"
    assert abs(diff["router"]["accuracy"]["delta"] - 0.05) < 1e-9
    assert abs(diff["router"]["macro_f1"]["delta"] - 0.06) < 1e-9
    assert abs(diff["router"]["per_class_f1"]["CODING"]["delta"] - 0.14) < 1e-9
    text = format_diff(diff)
    assert "accuracy" in text
    assert "+" in text


def test_diff_router_regression():
    a = _router_fixture(accuracy=0.95, macro_f1=0.94, coding_f1=0.99)
    b = _router_fixture(accuracy=0.90, macro_f1=0.88, coding_f1=0.85)
    diff = diff_experiments(a, b)
    assert diff["router"]["accuracy"]["delta"] < 0
    text = format_diff(diff)
    assert "accuracy" in text


def test_diff_pipeline_metrics():
    a = _pipeline_fixture(
        pass_rate=0.5,
        router_acc=0.5,
        mean_ms=2000.0,
        p95_ms=3000.0,
        coding_pass=0.0,
    )
    b = _pipeline_fixture(
        pass_rate=0.75,
        router_acc=0.75,
        mean_ms=1500.0,
        p95_ms=2500.0,
        coding_pass=1.0,
    )
    diff = diff_experiments(a, b)
    assert diff["kind"] == "pipeline"
    assert abs(diff["pipeline"]["pass_rate"]["delta"] - 0.25) < 1e-9
    assert abs(diff["pipeline"]["router_accuracy"]["delta"] - 0.25) < 1e-9
    assert abs(diff["pipeline"]["latency_mean_ms"]["delta"] - (-500.0)) < 1e-9
    assert abs(diff["pipeline"]["latency_p95_ms"]["delta"] - (-500.0)) < 1e-9
    assert abs(diff["pipeline"]["per_domain_pass_rate"]["CODING"]["delta"] - 1.0) < 1e-9
    text = format_diff(diff)
    assert "pass_rate" in text
    assert "per-domain PASS rate" in text


def test_format_orchestration_comparison():
    from arcs.eval.compare import format_orchestration_comparison, pass_stats

    naive = {
        "name": "naive-baseline-v1",
        "kind": "naive_baseline",
        "pipeline": {
            "n": 33,
            "status_counts": {"PASS": 9, "FAIL": 24, "ERROR": 0},
        },
    }
    arcs = {
        "name": "post-fix-v2-merged",
        "kind": "pipeline",
        "pipeline": {
            "n": 48,
            "status_counts": {"PASS": 14, "FAIL": 19, "ERROR": 15},
        },
    }
    text = format_orchestration_comparison(naive, arcs)
    assert "naive-baseline-v1" in text
    assert "post-fix-v2-merged" in text
    assert "27.3%" in text
    assert "42.4%" in text
    assert "ARCS − naive" in text
    assert pass_stats(arcs)["pass_pct"] == pytest.approx(42.424, rel=1e-3)


def test_diff_tolerates_missing_fields():
    a = {"name": "a", "meta": {}}
    b = {"name": "b", "meta": {}, "router": {"accuracy": 1.0}}
    diff = diff_experiments(a, b)
    assert diff["router"]["accuracy"]["a"] is None
    assert diff["router"]["accuracy"]["b"] == 1.0
    assert diff["router"]["accuracy"]["delta"] is None
    # Should not raise.
    format_diff(diff)
