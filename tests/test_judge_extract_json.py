"""Tests for the hardened JSON extraction in arcs.verification.judge."""

from __future__ import annotations

import pytest

from arcs.verification import judge


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
