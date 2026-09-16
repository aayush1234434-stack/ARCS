"""Provider-neutral token usage helpers."""

from __future__ import annotations

from typing import Any, Iterable

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
COUNTER_KEYS = (*USAGE_KEYS, "api_calls")


def _read(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def response_usage(response: Any) -> dict[str, int]:
    """Extract common token counters from Groq/OpenAI-compatible responses."""
    usage = _read(response, "usage")
    if usage is None:
        return {}
    result: dict[str, int] = {"api_calls": 1}
    for key in USAGE_KEYS:
        raw = _read(usage, key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            result[key] = value
    return result


def combine_usage(items: Iterable[dict[str, Any] | None]) -> dict[str, int]:
    """Sum normalized usage dictionaries, ignoring missing counters."""
    total = {key: 0 for key in COUNTER_KEYS}
    seen = False
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in COUNTER_KEYS:
            try:
                value = int(item.get(key, 0))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                total[key] += value
                seen = seen or value > 0
    return total if seen else {}
