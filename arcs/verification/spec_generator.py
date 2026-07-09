"""
Spec generator: define what a correct answer must look like for a user query.

Usage:
    python spec_generator.py
    python spec_generator.py "What is the maximum safe dose of acetaminophen for adults?"
"""

import argparse
import json
import re
import sys

from arcs.clients.groq import get_client
from arcs import config

# Deliberately different model family from the Llama-based generator by default.
# Qwen is Alibaba's architecture — correlated blind spots with Meta's Llama are minimal.
MODEL = config.DEFAULT_SPEC_MODEL

SYSTEM_PROMPT = """You are a specification generator. Your job is not to answer questions — it is to define what a correct answer must look like.

You will receive a user query. Output a structured specification that a judge can use to evaluate whether any given answer to that query is correct, complete, and appropriate.

Be precise and testable. Every criterion must be checkable by reading the answer — no domain expertise required.

IMPORTANT: Write real content tailored to the user's query. Never copy placeholder text or template labels.

Use exactly these section headers and fill each with query-specific content:

INTENT:
(one sentence describing what the user is trying to accomplish)

REQUIRED ELEMENTS:
- (specific thing that must appear in a correct answer)
- (add at least two more as needed)

CORRECTNESS CRITERIA:
- (specific checkable condition a correct answer satisfies)
- (add at least one more as needed)

DISQUALIFYING CONDITIONS:
- (specific thing that makes the answer wrong or incomplete)
- (add at least one more as needed)

SCOPE:
(what the answer should and should not cover; note if the question is ambiguous or cross-domain)"""

TEMPLATE_MARKERS = (
    "<one sentence",
    "<specific thing",
    "<required element",
    "<specific checkable",
    "<criterion 2>",
    "<specific thing that, if present",
    "<disqualifying condition",
    "<what the answer should",
    "your query here",
)


def _normalize_text(text: str) -> str:
    """Fix common spacing glitches in model output."""
    text = re.sub(r"per(\d)", r"per \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_bullets(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = re.sub(r"^\([a-zA-Z]\)\s+", "", line)
        line = _normalize_text(line)
        if line:
            items.append(line)
    return items


def _looks_like_template(parsed: dict) -> bool:
    blob = " ".join(
        [
            parsed.get("intent", ""),
            *parsed.get("required_elements", []),
            *parsed.get("correctness_criteria", []),
            *parsed.get("disqualifying_conditions", []),
            parsed.get("scope", ""),
        ]
    ).lower()
    return any(marker in blob for marker in TEMPLATE_MARKERS)


def parse_response(raw: str, model: str) -> dict:
    sections: dict[str, str] = {}
    keys = ["INTENT", "REQUIRED ELEMENTS", "CORRECTNESS CRITERIA", "DISQUALIFYING CONDITIONS", "SCOPE"]

    for i, key in enumerate(keys):
        start = raw.find(f"{key}:")
        if start == -1:
            sections[key.lower()] = ""
            continue
        start += len(f"{key}:")
        end = len(raw)
        for next_key in keys[i + 1:]:
            pos = raw.find(f"{next_key}:", start)
            if pos != -1:
                end = pos
                break
        sections[key.lower()] = raw[start:end].strip()

    return {
        "intent": _normalize_text(sections.get("intent", "")),
        "required_elements": _parse_bullets(sections.get("required elements", "")),
        "correctness_criteria": _parse_bullets(sections.get("correctness criteria", "")),
        "disqualifying_conditions": _parse_bullets(sections.get("disqualifying conditions", "")),
        "scope": _normalize_text(sections.get("scope", "")),
        "model": model,
    }


def _call_model(query: str, retry: bool = False) -> tuple[str, str]:
    user_content = query
    if retry:
        user_content = (
            f"{query}\n\n"
            "Reminder: replace every section with real, query-specific content. "
            "Do not output template placeholders."
        )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    return raw, response.model


def run(query: str) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    raw, model = _call_model(query)
    parsed = parse_response(raw, model=model)

    if _looks_like_template(parsed):
        raw, model = _call_model(query, retry=True)
        parsed = parse_response(raw, model=model)

    if _looks_like_template(parsed):
        raise ValueError(
            "Model returned template placeholders instead of a real spec. "
            "Try a specific user question."
        )

    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer specification for a user query."
    )
    parser.add_argument("query", nargs="?", help="User query (prompted if omitted)")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter your query: ").strip()
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            sys.exit(1)

    try:
        result = run(query)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
