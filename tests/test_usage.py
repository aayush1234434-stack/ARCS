from __future__ import annotations

from types import SimpleNamespace

from arcs.clients.usage import combine_usage, response_usage


def test_response_usage_normalizes_provider_object():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert response_usage(response) == {
        "api_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_combine_usage_sums_attempts():
    combined = combine_usage(
        [
            {"api_calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"api_calls": 1, "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        ]
    )
    assert combined["api_calls"] == 2
    assert combined["total_tokens"] == 27
