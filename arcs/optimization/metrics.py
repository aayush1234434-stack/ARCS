"""
Metrics for DSPy optimization — score answers with the existing LLM judge
or (for CODING) the sandbox verifier.

Sandbox verification is N/A for MEDICAL (non-executable). Use judge PASS/score.
CODING prefers sandbox PASS when test cases are available, else judge.
"""

from __future__ import annotations

from typing import Any

from arcs.pipelines.specialists.common import extract_code_block
from arcs.verification import judge, sandbox


def _example_field(example: Any, key: str, default: Any = None) -> Any:
    if hasattr(example, "get"):
        value = example.get(key, default)
        if value is not None:
            return value
    return getattr(example, key, default)


def _prediction_answer(prediction: Any) -> str:
    if prediction is None:
        return ""
    if isinstance(prediction, str):
        return prediction
    answer = getattr(prediction, "answer", None)
    if answer is not None:
        return str(answer)
    if hasattr(prediction, "get"):
        return str(prediction.get("answer", "") or "")
    return str(prediction)


def judge_score(
    example: Any,
    prediction: Any,
    *,
    trace: Any = None,
) -> float:
    """Return judge score in [0.0, 1.0] for a predicted answer.

    Uses ``example.specification`` (or ``example.spec``) and ``example.query``
    (or ``example.question``). On judge errors, returns 0.0.
    """
    del trace  # unused; DSPy may pass a trace
    question = str(_example_field(example, "query") or _example_field(example, "question") or "")
    specification = _example_field(example, "specification") or _example_field(example, "spec") or {}
    answer = _prediction_answer(prediction).strip()

    if not question.strip() or not answer:
        return 0.0
    if not isinstance(specification, dict) or not specification:
        return 0.0

    try:
        result = judge.run(
            question=question,
            answer=answer,
            specification=specification,
        )
    except Exception:
        return 0.0

    score = result.get("score")
    try:
        return float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def judge_pass(
    example: Any,
    prediction: Any,
    *,
    trace: Any = None,
) -> bool:
    """Binary metric: True when the judge returns verdict PASS.

    Proxy for "would not get NEGATIVE" — a passing judge score is treated as
    a successful specialist answer for optimization.
    """
    del trace
    question = str(_example_field(example, "query") or _example_field(example, "question") or "")
    specification = _example_field(example, "specification") or _example_field(example, "spec") or {}
    answer = _prediction_answer(prediction).strip()

    if not question.strip() or not answer:
        return False
    if not isinstance(specification, dict) or not specification:
        return False

    try:
        result = judge.run(
            question=question,
            answer=answer,
            specification=specification,
        )
    except Exception:
        return False

    return str(result.get("verdict", "")).upper() == "PASS"


# Alias used by DSPy optimizers (metric(example, prediction, trace=None)).
def judge_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """DSPy-compatible metric: judge PASS == success."""
    return judge_pass(example, prediction, trace=trace)


def sandbox_pass(
    example: Any,
    prediction: Any,
    *,
    trace: Any = None,
) -> bool:
    """Binary metric: True when sandbox.run returns verdict PASS.

    Uses ``example.test_cases`` (or ``example.tests``) and extracts code from
    the predicted answer via ``extract_code_block``.
    """
    del trace
    answer = _prediction_answer(prediction).strip()
    if not answer:
        return False

    test_cases = _example_field(example, "test_cases") or _example_field(example, "tests")
    if not isinstance(test_cases, list) or not test_cases:
        return False

    code = extract_code_block(answer).strip() or answer
    if not code.strip():
        return False

    try:
        result = sandbox.run(code, test_cases)
    except Exception:
        return False

    return str(result.get("verdict", "")).upper() == "PASS"


def coding_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """DSPy metric for CODING specialist optimization.

    Prefer sandbox PASS when the example has test cases. Otherwise fall back
    to the LLM judge when a specification is present. Returns False if neither
    path can evaluate the prediction.
    """
    del trace
    test_cases = _example_field(example, "test_cases") or _example_field(example, "tests")
    if isinstance(test_cases, list) and test_cases:
        return sandbox_pass(example, prediction)

    specification = _example_field(example, "specification") or _example_field(example, "spec")
    if isinstance(specification, dict) and specification:
        return judge_pass(example, prediction)

    return False


def _parse_verifier_prediction(prediction: Any) -> dict[str, Any]:
    """Normalize a DSPy judge prediction into a verdict/score dict."""
    raw = ""
    if prediction is None:
        raw = ""
    elif isinstance(prediction, str):
        raw = prediction
    elif hasattr(prediction, "verification_json"):
        raw = str(getattr(prediction, "verification_json") or "")
    elif hasattr(prediction, "answer"):
        raw = str(getattr(prediction, "answer") or "")
    elif hasattr(prediction, "get"):
        raw = str(
            prediction.get("verification_json")
            or prediction.get("answer")
            or prediction.get("verdict")
            or ""
        )
    else:
        raw = str(prediction)

    raw = raw.strip()
    if not raw:
        return {"verdict": "FAIL", "score": 0.0}

    # Already structured?
    if hasattr(prediction, "verdict") and getattr(prediction, "verdict", None):
        try:
            score = float(getattr(prediction, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return {
            "verdict": str(getattr(prediction, "verdict")).strip().upper(),
            "score": score,
        }

    try:
        parsed = judge._extract_json(raw)
        return judge._normalize_result(parsed)
    except Exception:
        # Soft parse: look for FAIL/PASS tokens
        upper = raw.upper()
        if '"VERDICT": "FAIL"' in upper or '"VERDICT":"FAIL"' in upper or "FAIL" in upper:
            return {"verdict": "FAIL", "score": 0.0}
        if '"VERDICT": "PASS"' in upper or '"VERDICT":"PASS"' in upper:
            return {"verdict": "PASS", "score": 0.9}
        return {"verdict": "FAIL", "score": 0.0}


def verifier_false_pass_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
) -> bool:
    """Metric for verifier prompt repair on known-bad answers.

    Training examples are VERIFIER-attributed failures: the user said NEGATIVE
    but the original judge passed (or scored high). Success means the
    *optimized* judge now returns FAIL or score < 0.75.
    """
    del trace
    expected = str(
        _example_field(example, "expected_verdict")
        or _example_field(example, "target_verdict")
        or "FAIL"
    ).strip().upper()

    result = _parse_verifier_prediction(prediction)
    verdict = str(result.get("verdict", "FAIL")).strip().upper()
    try:
        score = float(result.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    if expected == "FAIL":
        # Known-bad answer: optimized judge should reject it.
        return verdict == "FAIL" or score < 0.75

    if expected == "PASS":
        return verdict == "PASS" and score >= 0.75

    return verdict == expected


def _parse_spec_prediction(prediction: Any) -> dict[str, Any]:
    """Normalize a DSPy spec-generator prediction into a specification dict."""
    from arcs.verification import spec_generator

    raw = ""
    if prediction is None:
        raw = ""
    elif isinstance(prediction, str):
        raw = prediction
    elif hasattr(prediction, "specification"):
        value = getattr(prediction, "specification")
        if isinstance(value, dict):
            return value
        raw = str(value or "")
    elif hasattr(prediction, "answer"):
        raw = str(getattr(prediction, "answer") or "")
    elif hasattr(prediction, "get"):
        value = prediction.get("specification") or prediction.get("answer") or ""
        if isinstance(value, dict):
            return value
        raw = str(value or "")
    else:
        raw = str(prediction)

    raw = raw.strip()
    if not raw:
        return {}

    try:
        return spec_generator.parse_response(raw, model="dspy")
    except Exception:
        return {}


def spec_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """Experimental metric for spec-generator prompt repair.

    Success when the predicted specification has **more** ``required_elements``
    than the logged baseline *and* the LLM judge scores the same fixed
    ``(query, answer)`` pair higher under the new spec.
    """
    del trace
    question = str(
        _example_field(example, "query") or _example_field(example, "question") or ""
    ).strip()
    answer = str(
        _example_field(example, "answer")
        or _example_field(example, "prior_answer")
        or ""
    ).strip()
    if not question or not answer:
        return False

    try:
        baseline_count = int(
            _example_field(example, "baseline_required_count")
            or _example_field(example, "required_count")
            or 0
        )
    except (TypeError, ValueError):
        baseline_count = 0

    try:
        baseline_score = float(
            _example_field(example, "baseline_judge_score")
            or _example_field(example, "prior_judge_score")
            or 0.0
        )
    except (TypeError, ValueError):
        baseline_score = 0.0

    new_spec = _parse_spec_prediction(prediction)
    if not isinstance(new_spec, dict) or not new_spec:
        return False

    required = new_spec.get("required_elements") or []
    if not isinstance(required, list):
        return False
    new_count = len([item for item in required if str(item).strip()])
    if new_count <= baseline_count:
        return False

    try:
        result = judge.run(
            question=question,
            answer=answer,
            specification=new_spec,
        )
        new_score = float(result.get("score") or 0.0)
    except Exception:
        return False

    return new_score > baseline_score
