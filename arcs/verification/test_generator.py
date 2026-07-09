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
    """Parse a JSON array from model output (tolerates fences and trailing text)."""
    text = raw.strip()
    if not text:
        raise ValueError("Test generator returned empty response")

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("[")
        if start == -1:
            raise ValueError(f"Test generator response is not a JSON list: {raw[:200]}")
        depth = 0
        end = -1
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            raise ValueError(f"Test generator response has unclosed JSON list: {raw[:200]}")
        text = text[start : end + 1]

    try:
        parsed, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Test generator response is not valid JSON: {raw[:200]}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Test generator response must be a JSON list, got {type(parsed).__name__}")
    return parsed


def _coerce_snippet(item) -> str:
    """Best-effort extraction of a test snippet from a model-produced item.

    Tolerates the shapes models actually emit: plain strings, dicts keyed by
    ``code``/``assert``/``test``, and nested lists of lines (which we join).
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        value = item.get("code") or item.get("assert") or item.get("test")
        if isinstance(value, (list, tuple)):
            return _coerce_snippet(list(value))
        return str(value or "").strip()
    if isinstance(item, (list, tuple)):
        lines = [_coerce_snippet(sub) for sub in item]
        return "\n".join(line for line in lines if line).strip()
    if item is None:
        return ""
    return str(item).strip()


def _normalize_tests(data: list) -> list[str]:
    tests: list[str] = []
    for item in data:
        snippet = _coerce_snippet(item)
        if snippet:
            tests.append(snippet)
    if not tests:
        raise ValueError("test generator returned no usable test snippets")
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
    try:
        parsed = _extract_json_list(raw)
    except ValueError:
        # One retry with stricter reminder (models sometimes add prose after JSON).
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
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your last reply was not parseable as a JSON list only. "
                        "Reply again with ONLY a JSON array of strings — no markdown, "
                        "no explanation before or after."
                    ),
                },
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        parsed = _extract_json_list(raw)
    tests = _normalize_tests(parsed)

    return {
        "test_cases": tests,
        "model": response.model,
        "count": len(tests),
    }
