"""Tests for strict vs relaxed judge verdict policy (JUDGE_STRICT env)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from arcs.verification import judge


def _base_result(**overrides) -> dict:
    data = {
        "verdict": "FAIL",
        "score": 0.8,
        "missing_required_elements": [],
        "incorrect_claims": [],
        "unsupported_claims": [],
        "disqualifying_conditions_triggered": [],
        "explanation": "ok",
    }
    data.update(overrides)
    return judge._normalize_result(data)


def test_strict_mode_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("JUDGE_STRICT", None)
        assert judge.is_strict_mode() is True


def test_relaxed_mode_via_env():
    with patch.dict(os.environ, {"JUDGE_STRICT": "0"}):
        assert judge.is_strict_mode() is False


@pytest.mark.parametrize("value", ["false", "no", "off", "FALSE"])
def test_relaxed_mode_truthy_strings(value: str):
    with patch.dict(os.environ, {"JUDGE_STRICT": value}):
        assert judge.is_strict_mode() is False


def test_strict_pass_when_complete():
    result = judge.apply_verdict_policy(
        {
            "score": 0.85,
            "missing_required_elements": [],
            "incorrect_claims": [],
            "disqualifying_conditions_triggered": [],
        },
        strict=True,
    )
    assert result["verdict"] == "PASS"


def test_strict_fail_on_one_missing():
    result = judge.apply_verdict_policy(
        {
            "score": 0.85,
            "missing_required_elements": ["Monitoring plan"],
            "incorrect_claims": [],
            "disqualifying_conditions_triggered": [],
        },
        strict=True,
    )
    assert result["verdict"] == "FAIL"


def test_relaxed_pass_on_one_missing():
    result = judge.apply_verdict_policy(
        {
            "score": 0.85,
            "missing_required_elements": ["Monitoring plan"],
            "incorrect_claims": [],
            "disqualifying_conditions_triggered": [],
        },
        strict=False,
    )
    assert result["verdict"] == "PASS"


def test_relaxed_fail_on_two_missing():
    result = judge.apply_verdict_policy(
        {
            "score": 0.85,
            "missing_required_elements": ["Workup", "Monitoring"],
            "incorrect_claims": [],
            "disqualifying_conditions_triggered": [],
        },
        strict=False,
    )
    assert result["verdict"] == "FAIL"


def test_both_modes_fail_on_incorrect_claims():
    payload = {
        "score": 0.9,
        "missing_required_elements": [],
        "incorrect_claims": ["Wrong dose"],
        "disqualifying_conditions_triggered": [],
    }
    assert judge.apply_verdict_policy(payload, strict=True)["verdict"] == "FAIL"
    assert judge.apply_verdict_policy(payload, strict=False)["verdict"] == "FAIL"


def test_both_modes_fail_below_score_threshold():
    payload = {
        "score": 0.74,
        "missing_required_elements": [],
        "incorrect_claims": [],
        "disqualifying_conditions_triggered": [],
    }
    assert judge.apply_verdict_policy(payload, strict=True)["verdict"] == "FAIL"
    assert judge.apply_verdict_policy(payload, strict=False)["verdict"] == "FAIL"


def test_normalize_result_uses_env_strict_by_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("JUDGE_STRICT", None)
        result = judge._normalize_result(
            {
                "verdict": "PASS",
                "score": 0.8,
                "missing_required_elements": ["One gap"],
            }
        )
    assert result["verdict"] == "FAIL"


def test_normalize_result_relaxed_env():
    with patch.dict(os.environ, {"JUDGE_STRICT": "0"}):
        result = judge._normalize_result(
            {
                "verdict": "PASS",
                "score": 0.8,
                "missing_required_elements": ["One gap"],
            }
        )
    assert result["verdict"] == "PASS"
