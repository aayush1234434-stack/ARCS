"""Tests for the hardened JSON extraction in arcs.verification.judge."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from arcs.verification import judge

EVAL_021_MALFORMED = (
    "<think>Compare tenant rights against required elements...</think>\n"
    "{\n"
    '  "verification_type": "LLM_JUDGE",\n'
    '  "verdict": "FAIL",\n'
    '  "score": 0.5,\n'
    '  "missing_required_elements": [\n'
    '    "Tenant\'s right to quiet enjoyment under California law",\n'
    '    "Statutory notice requirements under Cal. Civ. Code § 1954"\n'
    "  ],\n"
    '  "incorrect_claims": [],\n'
    '  "unsupported_claims": [],\n'
    '  "disqualifying_conditions_triggered": [],\n'
    '  "explanation": "The answer mentions "quiet enjoyment" but omits the statute"\n'
    "}"
)

MINIMAL_SPEC = {
    "intent": "Explain tenant rights when a landlord enters without notice.",
    "required_elements": ["Quiet enjoyment", "Notice requirements"],
    "correctness_criteria": ["Cites California law"],
    "disqualifying_conditions": ["Fabricated statutes"],
    "scope": "California residential tenancy only.",
}


def test_plain_json_object():
    raw = '{"verdict": "PASS", "score": 0.9}'
    assert judge._extract_json(raw) == {"verdict": "PASS", "score": 0.9}


def test_strips_think_block():
    raw = (
        "<think>Let me reason about this at length...\n"
        "still reasoning</think>\n"
        '{"verdict": "FAIL", "score": 0.2}'
    )
    assert judge._extract_json(raw) == {"verdict": "FAIL", "score": 0.2}


def test_fenced_json():
    raw = '```json\n{"verdict": "PASS", "score": 1.0}\n```'
    assert judge._extract_json(raw) == {"verdict": "PASS", "score": 1.0}


def test_leading_and_trailing_prose():
    raw = (
        "Here is my assessment of the answer:\n"
        '{"verdict": "FAIL", "score": 0.5, "explanation": "missing detail"}\n'
        "I hope this helps!"
    )
    result = judge._extract_json(raw)
    assert result["verdict"] == "FAIL"
    assert result["explanation"] == "missing detail"


def test_nested_object_bracket_match():
    raw = (
        'prefix {"verdict": "PASS", "score": 0.8, '
        '"nested": {"a": 1, "b": [2, 3]}} trailing text'
    )
    result = judge._extract_json(raw)
    assert result["nested"] == {"a": 1, "b": [2, 3]}


def test_trailing_comma_repair():
    raw = '{"verdict": "FAIL", "score": 0.0, "missing_required_elements": [],}'
    result = judge._extract_json(raw)
    assert result["verdict"] == "FAIL"


def test_braces_inside_string_do_not_break_matching():
    raw = '{"verdict": "PASS", "explanation": "use a dict like {x: 1}", "score": 0.9}'
    result = judge._extract_json(raw)
    assert result["explanation"] == "use a dict like {x: 1}"
    assert result["score"] == 0.9


def test_unparseable_raises_valueerror():
    with pytest.raises(ValueError):
        judge._extract_json("no json here at all")


def test_empty_after_think_raises():
    with pytest.raises(ValueError):
        judge._extract_json("<think>only reasoning, no answer</think>")


def test_eval_021_pattern_unescaped_quotes_still_unparseable():
    """eval-021 hit JSONDecodeError: Expecting ',' delimiter (unescaped quotes)."""
    with pytest.raises(ValueError):
        judge._extract_json(EVAL_021_MALFORMED)


def test_eval_021_pattern_thinking_block_stripped_before_parse_attempt():
    """Thinking is removed; remaining body is still invalid JSON."""
    stripped = judge._strip_reasoning(EVAL_021_MALFORMED)
    assert "<think>" not in stripped
    assert stripped.lstrip().startswith("{")
    with pytest.raises(ValueError):
        judge._extract_json(stripped)


def test_run_returns_normalized_fail_after_retry_on_eval_021_pattern():
    """run() must not raise into eval_pipeline when parsing keeps failing."""
    with patch.object(judge, "_call_judge", side_effect=ValueError("not json")):
        result = judge.run(
            question="What rights does a tenant have if the landlord enters without notice in California?",
            answer="Tenants have a right to quiet enjoyment.",
            specification=MINIMAL_SPEC,
            model="test-model",
        )
    assert result["verdict"] == "FAIL"
    assert result["score"] == 0.0
    assert result["explanation"] == "judge parse error"
    assert result["missing_required_elements"] == []
    assert result["model"] == "test-model"
