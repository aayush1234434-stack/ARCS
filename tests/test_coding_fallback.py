"""Tests for coding-path judge fallback (prose-only / non-Python answers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arcs.main import (
    _effective_specialist_answer,
    _run_sandbox_pipeline,
    _should_defer_to_judge,
)
from arcs.pipelines.registry import Pipeline


def _coding_pipeline() -> Pipeline:
    from arcs.pipelines.specialists import coding

    return Pipeline(
        domain="CODING",
        verifier="sandbox",
        specialist=coding,
        tools=("test_generator",),
        max_retries=2,
    )


def _mock_components(*, sandbox_run: MagicMock | None = None) -> dict:
    sandbox = MagicMock()
    sandbox.run = sandbox_run or MagicMock(
        return_value={"verdict": "PASS", "score": 1.0, "verification_type": "SANDBOX"}
    )
    return {
        "test_generator": MagicMock(
            return_value={"test_cases": ["assert True"], "model": "test-gen"}
        ),
        "sandbox": sandbox,
    }


@patch("arcs.main.ThreadPoolExecutor")
def test_prose_only_answer_defers_to_judge(mock_pool):
    """eval-042 style: plain-English walkthrough with no code fences."""
    prose = (
        "EXPLANATION:\n"
        "A memoizing decorator caches function results keyed by arguments.\n"
        "Use functools.lru_cache or build a dict keyed by args."
    )
    mock_pool.return_value.__enter__.return_value.submit.side_effect = [
        MagicMock(result=MagicMock(return_value={"test_cases": [], "model": "tg"})),
        MagicMock(
            result=MagicMock(
                return_value={
                    "answer": "",
                    "explanation": prose,
                    "pipeline_id": "CODING",
                }
            )
        ),
    ]

    pipeline = _coding_pipeline()
    components = _mock_components()

    specialist, verification, tooling = _run_sandbox_pipeline(
        query="In plain English, walk me through writing a memoizing decorator",
        pipeline=pipeline,
        specification={},
        components=components,
    )

    assert tooling.get("verifier_fallback") == "judge"
    assert verification == {}
    components["sandbox"].run.assert_not_called()


@patch("arcs.main.ThreadPoolExecutor")
def test_non_python_fence_defers_to_judge(mock_pool):
    js_answer = "Use debounce:\n```javascript\nfunction debounce(fn, ms) {}\n```"
    mock_pool.return_value.__enter__.return_value.submit.side_effect = [
        MagicMock(result=MagicMock(return_value={"test_cases": [], "model": "tg"})),
        MagicMock(
            result=MagicMock(
                return_value={"answer": js_answer, "pipeline_id": "CODING"}
            )
        ),
    ]

    pipeline = _coding_pipeline()
    components = _mock_components()

    _, verification, tooling = _run_sandbox_pipeline(
        query="Explain debounce in JavaScript",
        pipeline=pipeline,
        specification={},
        components=components,
    )

    assert tooling.get("verifier_fallback") == "judge"
    assert verification == {}
    components["sandbox"].run.assert_not_called()


@patch("arcs.main.ThreadPoolExecutor")
def test_python_code_runs_sandbox(mock_pool):
    py_answer = "SOLUTION:\n```python\ndef add(a, b):\n    return a + b\n```"
    mock_pool.return_value.__enter__.return_value.submit.side_effect = [
        MagicMock(result=MagicMock(return_value={"test_cases": ["assert add(1,2)==3"], "model": "tg"})),
        MagicMock(
            result=MagicMock(
                return_value={"answer": py_answer, "pipeline_id": "CODING"}
            )
        ),
    ]

    pipeline = _coding_pipeline()
    sandbox_run = MagicMock(
        return_value={"verdict": "PASS", "score": 1.0, "verification_type": "SANDBOX"}
    )
    components = _mock_components(sandbox_run=sandbox_run)

    _, verification, tooling = _run_sandbox_pipeline(
        query="Write add(a,b)",
        pipeline=pipeline,
        specification={},
        components=components,
    )

    assert tooling.get("verifier_fallback") is None
    assert verification.get("verdict") == "PASS"
    sandbox_run.assert_called_once()


def test_effective_specialist_answer_falls_back_to_explanation():
    result = {
        "answer": "",
        "explanation": "Step one: define a wrapper that stores a cache dict.",
    }
    assert _effective_specialist_answer(result) == (
        "Step one: define a wrapper that stores a cache dict."
    )


def test_should_defer_to_judge_for_empty_python_fence():
    assert _should_defer_to_judge({"answer": "```python\n\n```"}) is True


def test_should_not_defer_when_runnable_python_present():
    assert _should_defer_to_judge(
        {"answer": "```python\ndef f():\n    return 1\n```"}
    ) is False
