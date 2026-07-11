"""Unit tests for arcs.clients.rate_limit — no network calls."""

from __future__ import annotations

import pytest

from arcs.clients.rate_limit import (
    is_groq_tpd_exhausted,
    is_rate_limit,
    parse_retry_after_seconds,
)

# Realistic Groq TPD message (eval-044 / post-fix-full-v1 pattern).
_GROQ_TPD_MSG = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_test` service tier "
    "`on_demand` on tokens per day (TPD): Limit 100000, Used 99409, "
    "Requested 996. Please try again in 5m49.919999999s.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)

# TPM-only 429 (no daily limit wording).
_GROQ_TPM_MSG = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` on tokens per minute (TPM): Limit 6000, "
    "Used 5990, Requested 500.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


class _MockExc(Exception):
    """Minimal stand-in for Groq/OpenAI SDK exceptions."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_is_groq_tpd_exhausted_plain_exception_string():
    assert is_groq_tpd_exhausted(Exception(_GROQ_TPD_MSG)) is True


def test_is_groq_tpd_exhausted_with_status_code_429():
    exc = _MockExc("tokens per day (TPD) limit hit", status_code=429)
    assert is_groq_tpd_exhausted(exc) is True


def test_is_groq_tpd_exhausted_with_body_dict():
    exc = _MockExc(
        "rate limit",
        status_code=429,
        body={"error": {"message": "Limit on tokens per day (TPD) exceeded"}},
    )
    assert is_groq_tpd_exhausted(exc) is True


def test_is_groq_tpd_exhausted_tpd_marker_case_insensitive():
    exc = _MockExc("Error code: 429 - TPD quota exhausted for org")
    assert is_groq_tpd_exhausted(exc) is True


def test_is_groq_tpd_exhausted_false_for_tpm_only():
    assert is_groq_tpd_exhausted(Exception(_GROQ_TPM_MSG)) is False


def test_is_groq_tpd_exhausted_false_for_non_rate_limit():
    assert is_groq_tpd_exhausted(Exception("connection reset")) is False


def test_is_groq_tpd_exhausted_false_for_tpd_without_429():
    assert is_groq_tpd_exhausted(Exception("tokens per day limit (internal)")) is False


def test_is_rate_limit_detects_429_status():
    assert is_rate_limit(_MockExc("x", status_code=429)) is True


def test_parse_retry_after_seconds_from_message():
    exc = Exception(
        "Error code: 429 - Rate limit on tokens per day (TPD). "
        "Please try again in 349.92s."
    )
    assert parse_retry_after_seconds(exc) == pytest.approx(349.92, rel=0.01)
