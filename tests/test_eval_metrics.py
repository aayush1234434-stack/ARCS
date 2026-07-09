"""Unit tests for arcs.eval.metrics — no API calls."""

from __future__ import annotations

from arcs.eval.metrics import aggregate_experiment, pipeline_summary, router_accuracy


def test_router_accuracy_basic():
    rows = [
        {"predicted_domain": "CODING", "expected_domain": "CODING"},
        {"predicted_domain": "MEDICAL", "expected_domain": "CODING"},
        {"predicted_domain": "LEGAL", "expected_domain": "LEGAL"},
        {"predicted_domain": "general", "expected_domain": "GENERAL"},
    ]
    result = router_accuracy(rows)
    assert result["n"] == 4
    assert result["correct"] == 3
    assert result["accuracy"] == 0.75
    assert result["per_domain_n"]["CODING"] == 2
    assert result["per_domain_accuracy"]["CODING"] == 0.5
    assert result["per_domain_accuracy"]["LEGAL"] == 1.0
    assert result["confusion"]["CODING"]["MEDICAL"] == 1
    assert result["confusion"]["CODING"]["CODING"] == 1
    assert result["skipped"] == 0


def test_router_accuracy_skips_incomplete():
    rows = [
        {"predicted_domain": "CODING"},  # missing expected
        {"expected_domain": "MEDICAL"},  # missing predicted
        "not-a-dict",
        {"predicted_domain": "LEGAL", "expected_domain": "LEGAL"},
    ]
    result = router_accuracy(rows)
    assert result["n"] == 1
    assert result["correct"] == 1
    assert result["skipped"] == 3
    assert result["accuracy"] == 1.0


def test_router_accuracy_empty():
    result = router_accuracy([])
    assert result["n"] == 0
    assert result["accuracy"] is None
    assert result["correct"] == 0


def test_pipeline_summary_status_and_latency():
    rows = [
        {
            "status": "PASS",
            "expected_domain": "CODING",
            "verification": {"verdict": "PASS"},
            "timing": {
                "total_ms": 1000,
                "route_ms": 10,
                "specialist_ms": 800,
                "verification_ms": 190,
            },
        },
        {
            "status": "FAIL",
            "expected_domain": "MEDICAL",
            "verification": {"verdict": "FAIL"},
            "timing": {
                "total_ms": 2000,
                "route_ms": 20,
                "specialist_ms": 1500,
                "verification_ms": 480,
            },
        },
        {
            "status": "UNKNOWN",
            "expected_domain": "LEGAL",
            "verification": {},
            "timing": {"total_ms": 1500},
        },
        {
            "error": "boom",
            "expected_domain": "GENERAL",
            "timing": {"total_ms": 50},
        },
    ]
    result = pipeline_summary(rows)
    assert result["n"] == 4
    assert result["status_counts"]["PASS"] == 1
    assert result["status_counts"]["FAIL"] == 1
    assert result["status_counts"]["UNKNOWN"] == 1
    assert result["error_count"] == 1
    assert result["status_rates"]["PASS"] == 0.25
    assert result["verdict_counts"]["PASS"] == 1
    assert result["verdict_counts"]["FAIL"] == 1
    assert result["per_domain"]["CODING"]["n"] == 1
    assert result["per_domain"]["CODING"]["pass_rate"] == 1.0

    total = result["latency_ms"]["total_ms"]
    assert total["count"] == 4
    assert total["mean"] == 1137.5
    assert total["p50"] == 1500.0
    assert total["p95"] == 2000.0


def test_pipeline_summary_empty():
    result = pipeline_summary([])
    assert result["n"] == 0
    assert result["status_rates"]["PASS"] is None
    assert result["latency_ms"]["total_ms"]["count"] == 0


def test_aggregate_experiment_shape():
    router = router_accuracy(
        [{"predicted_domain": "CODING", "expected_domain": "CODING"}]
    )
    pipeline = pipeline_summary([{"status": "PASS", "expected_domain": "CODING"}])
    result = aggregate_experiment(
        "baseline-v1",
        router=router,
        pipeline=pipeline,
        meta={"note": "unit-test"},
    )
    assert result["name"] == "baseline-v1"
    assert result["router"]["accuracy"] == 1.0
    assert result["pipeline"]["n"] == 1
    assert result["meta"]["note"] == "unit-test"
