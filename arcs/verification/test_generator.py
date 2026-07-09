"""
Independent test-case generator for the coding sandbox.

Uses a different model family from the coding generator by default so that
correlated failures (same bug in solution and tests) are less likely.
"""

from __future__ import annotations

import json
import re

from arcs.clients.groq import get_client
from arcs import config

SYSTEM_PROMPT = """You generate Python assert-based test snippets for verifying a coding solution.

Rules:
- Do NOT write the full solution. Only write short test snippets that will be exec()'d
  in the same Python namespace as the candidate solution.
- Each test must be a self-contained snippet using assert (or raising AssertionError).
- Assume the candidate solution defines the requested function(s) or symbols in the query.
- Cover typical cases and at least one edge case.
- Prefer 3–5 tests. Never invent APIs not implied by the query.
- Output ONLY valid JSON: a list of strings. No markdown, no commentary.

Example output:
[
  "assert reverse_string('abc') == 'cba'",
  "assert reverse_string('') == ''",
  "assert reverse_string('a') == 'a'"
]
"""


def _extract_json_list(raw: str) -> list:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Test generator response is not a JSON list: {raw[:200]}")
        text = text[start : end + 1]
    return json.loads(text)


def _normalize_tests(data: list) -> list[str]:
    tests: list[str] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            snippet = item.strip()
        elif isinstance(item, dict):
            snippet = str(
                item.get("code") or item.get("assert") or item.get("test") or ""
            ).strip()
        else:
            raise TypeError(f"test_cases[{index}] must be a str or dict")
        if not snippet:
            raise ValueError(f"test_cases[{index}] is empty")
        tests.append(snippet)
    if not tests:
        raise ValueError("test generator returned an empty list")
    return tests


def run(query: str, *, model: str | None = None) -> dict:
    """Generate independent Python assert snippets for a coding query."""
    model = model or config.DEFAULT_TEST_GENERATOR_MODEL
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate assert-based Python test snippets for this coding task:\n\n"
                    f"{query}"
                ),
            },
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_list(raw)
    tests = _normalize_tests(parsed)

    return {
        "test_cases": tests,
        "model": response.model,
        "count": len(tests),
    }
