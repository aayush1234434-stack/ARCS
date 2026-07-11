"""Helpers for detecting and reacting to Groq rate-limit errors.

Groq enforces both per-minute (TPM) and per-day (TPD) token limits. TPM limits
are transient and the SDK's built-in retries usually ride them out; TPD limits
are not recoverable within a run, so eval tooling should stop and let the user
resume later. These helpers classify the error without importing the Groq SDK.
"""

from __future__ import annotations

import re
from typing import Any

_TPD_MARKERS = (
    "tokens per day",
    "tpd",
    "per day",
)


def _status_code(exc: Any) -> int | None:
    """Best-effort HTTP status code from a Groq/OpenAI-style exception."""
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _message(exc: Any) -> str:
    """Collect searchable text from an exception (message + body)."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            msg = error.get("message")
            if msg:
                parts.append(str(msg))
        elif error:
            parts.append(str(error))
    return " ".join(parts).lower()


def is_rate_limit(exc: Any) -> bool:
    """True if the exception looks like an HTTP 429 rate-limit error."""
    if _status_code(exc) == 429:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    return "rate limit" in _message(exc) or "rate_limit" in _message(exc)


def is_groq_tpd_exhausted(exc: Any) -> bool:
    """True if the error is a 429 caused by the per-day token limit (TPD).

    Requires both a 429 signal (HTTP status, ``Error code: 429``, or rate-limit
    phrasing) **and** a TPD marker (``tokens per day`` or ``TPD``). TPM-only 429s
    are excluded so eval can keep retrying those within the same run.
    """
    message = _message(exc)
    if not any(marker in message for marker in _TPD_MARKERS):
        return False
    if _status_code(exc) == 429:
        return True
    if "error code: 429" in message:
        return True
    return is_rate_limit(exc)


def parse_retry_after_seconds(exc: Any) -> float | None:
    """Extract a suggested wait time (seconds) from a rate-limit error.

    Checks the ``Retry-After`` response header first, then falls back to the
    ``try again in 12.3s`` phrasing Groq embeds in the error message.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except AttributeError:
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    message = _message(exc)
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", message)
    if match:
        return float(match.group(1))
    return None
