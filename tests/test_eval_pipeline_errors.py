"""eval_pipeline error_class on ERROR rows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "eval_pipeline", _ROOT / "scripts" / "eval_pipeline.py"
)
eval_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_pipeline)

from arcs.main import PipelineError


def test_build_row_result_includes_error_class():
    row = {"id": "eval-001", "query": "q", "expected_domain": "CODING"}
    state = {
        "query_id": "abc-123",
        "route": {"domain": "CODING", "confidence": 0.99},
        "pipeline": {"pipeline_id": "CODING", "verifier": "sandbox"},
        "error": "Rate limit reached",
        "error_class": "rate_limit",
    }
    result = eval_pipeline._build_row_result(
        row,
        state,
        error=state["error"],
        error_class=state["error_class"],
    )
    assert result["status"] == "ERROR"
    assert result["error_class"] == "rate_limit"
    assert result["query_id"] == "abc-123"
    assert result["predicted_domain"] == "CODING"


def test_run_eval_pipeline_error(monkeypatch):
    state = {
        "query_id": "run-1",
        "query": "hello",
        "error": "429 Too Many Requests",
        "error_class": "rate_limit",
        "route": {},
        "timing": {},
    }

    def _fail(_query: str):
        raise PipelineError(state)

    monkeypatch.setattr(eval_pipeline, "run_pipeline", _fail)

    rows = [{"id": "eval-001", "query": "hello", "expected_domain": "GENERAL"}]
    results, counters, tpd = eval_pipeline.run_eval(rows, dry_run=False)
    assert len(results) == 1
    assert results[0]["error_class"] == "rate_limit"
    assert results[0]["query_id"] == "run-1"
    assert counters["errors"] == 1
