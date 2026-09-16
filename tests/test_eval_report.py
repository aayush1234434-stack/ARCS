from __future__ import annotations

from arcs.eval.metrics import pipeline_summary, wilson_interval
from arcs.eval.report import render_markdown


def test_wilson_interval_handles_small_samples():
    interval = wilson_interval(32, 48)
    assert interval["low"] < 32 / 48 < interval["high"]
    assert interval["n"] == 48


def test_pipeline_summary_includes_completed_confidence_interval():
    rows = [
        {"status": "PASS", "expected_domain": "GENERAL"},
        {"status": "FAIL", "expected_domain": "GENERAL"},
        {"status": "ERROR", "error": "rate limit", "expected_domain": "GENERAL"},
    ]
    summary = pipeline_summary(rows)
    assert summary["completed_n"] == 2
    assert summary["completed_pass_rate"] == 0.5
    assert summary["completed_pass_ci95"]["n"] == 2


def test_report_calls_out_human_evaluation_limit():
    experiment = {
        "name": "sample",
        "pipeline": {
            "status_counts": {"PASS": 3, "FAIL": 1, "ERROR": 0},
            "per_domain": {},
        },
        "meta": {"run_id": "run-1", "git_commit": "abc", "schema_version": "1.0"},
    }
    report = render_markdown(experiment, source_sha256="deadbeef")
    assert "75.0%" in report
    assert "95% Wilson CI" in report
    assert "blinded human evaluation" in report
    assert "deadbeef" in report
