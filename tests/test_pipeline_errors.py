"""Pipeline error classification and query_id preservation."""

from __future__ import annotations

import pytest

from arcs.main import PipelineError, classify_pipeline_error, run_pipeline


class _RateLimitError(Exception):
    status_code = 429

    def __str__(self) -> str:
        return "Rate limit reached for tokens per day (TPD)"


def test_classify_rate_limit():
    assert classify_pipeline_error(_RateLimitError()) == "rate_limit"


def test_classify_judge_parse():
    exc = ValueError("Judge response is not JSON: {broken")
    assert classify_pipeline_error(exc) == "judge_parse"


def test_classify_sandbox():
    exc = RuntimeError("Sandbox did not emit a result marker")
    assert classify_pipeline_error(exc) == "sandbox"


def test_classify_empty_code():
    exc = ValueError("code cannot be empty")
    assert classify_pipeline_error(exc) == "empty_code"


def test_classify_unknown():
    assert classify_pipeline_error(RuntimeError("something else")) == "unknown"


def test_run_pipeline_empty_query_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        run_pipeline("   ")


def test_pipeline_error_carries_state(monkeypatch):
    def _boom(_query: str):
        raise RuntimeError("router exploded")

    import arcs.router as router_mod

    monkeypatch.setattr(router_mod, "route", _boom)

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline("test query")

    state = exc_info.value.state
    assert state["query_id"]
    assert state["query"] == "test query"
    assert state["error"] == "router exploded"
    assert state["error_class"] == "unknown"
