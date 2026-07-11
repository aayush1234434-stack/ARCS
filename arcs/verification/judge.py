"""
Verifier — independent LLM judge for ARCS.

Compares a specialist answer against a specification checklist produced by
the Spec Generator. Does not know about routing or specialist selection.

Usage:
    python judge.py
    python judge.py --question "..." --answer "..." --spec spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv

from arcs import config

load_dotenv(config.PROJECT_ROOT / ".env")

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = config.DEFAULT_JUDGE_MODEL
TIMEOUT_SECONDS = 120

VERIFICATION_TYPE = "LLM_JUDGE"

RESULT_TEMPLATE: dict[str, Any] = {
    "verification_type": VERIFICATION_TYPE,
    "verdict": "FAIL",
    "score": 0.0,
    "missing_required_elements": [],
    "incorrect_claims": [],
    "unsupported_claims": [],
    "disqualifying_conditions_triggered": [],
    "explanation": "",
}

SPEC_KEYS = (
    "intent",
    "required_elements",
    "correctness_criteria",
    "disqualifying_conditions",
    "scope",
)

SYSTEM_PROMPT = """You are an independent answer verifier. You do not answer the user's question.

You receive:
1. The original user question
2. A specialist's answer to that question
3. A specification checklist that defines what a correct answer must contain

Your job is to compare the answer against the specification only. Do not use outside knowledge to invent requirements beyond the specification.

Evaluation rules:
- Check every item in required_elements — list any that are missing in missing_required_elements.
- Check correctness_criteria — put violated criteria in incorrect_claims (as short strings).
- Check disqualifying_conditions — if any are triggered, list them in disqualifying_conditions_triggered and set verdict to FAIL.
- Check scope — if the answer covers forbidden or irrelevant ground, note it in explanation and lower the score.
- unsupported_claims: factual claims in the answer that the specification required but are stated without adequate support, or claims that contradict the specification.

Scoring:
- score is a float from 0.0 to 1.0 (1.0 = fully satisfies the specification).
- verdict is PASS only if score >= 0.75 AND no disqualifying_conditions_triggered AND no missing_required_elements.

Return ONLY valid JSON with exactly these keys (no markdown, no extra keys):
{
    "verification_type": "LLM_JUDGE",
    "verdict": "PASS" or "FAIL",
    "score": 0.0,
    "missing_required_elements": [],
    "incorrect_claims": [],
    "unsupported_claims": [],
    "disqualifying_conditions_triggered": [],
    "explanation": ""
}"""

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client

    from openai import OpenAI

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set in .env")

    _client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT_SECONDS,
    )
    return _client


def _validate_specification(specification: dict) -> None:
    if not isinstance(specification, dict):
        raise TypeError("specification must be a dict")

    for key in SPEC_KEYS:
        if key not in specification:
            raise ValueError(f"specification missing required key: {key}")

    if not isinstance(specification["required_elements"], list):
        raise TypeError("specification.required_elements must be a list")
    if not isinstance(specification["correctness_criteria"], list):
        raise TypeError("specification.correctness_criteria must be a list")
    if not isinstance(specification["disqualifying_conditions"], list):
        raise TypeError("specification.disqualifying_conditions must be a list")


def _build_user_message(question: str, answer: str, specification: dict) -> str:
    spec_payload = {key: specification[key] for key in SPEC_KEYS}
    return (
        f"QUESTION:\n{question.strip()}\n\n"
        f"ANSWER:\n{answer.strip()}\n\n"
        f"SPECIFICATION:\n{json.dumps(spec_payload, indent=2)}"
    )


def _strip_reasoning(raw: str) -> str:
    """Drop <think>...</think> reasoning blocks that break JSON parsing."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if "<think>" in text:
        text = text[text.rfind("<think>") + len("<think>") :]
    return text.strip()


def _bracket_match(text: str, start: int) -> str | None:
    """Return the balanced ``{...}`` substring beginning at ``start``.

    Tracks string state so braces inside string literals do not affect depth.
    Returns None if no balanced object is found.
    """
    depth = 0
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
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (a common model JSON defect)."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _extract_json(raw: str) -> dict:
    text = _strip_reasoning(raw)
    if not text:
        raise ValueError("Judge returned empty response")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError(f"Judge response is not JSON: {raw[:200]}")

    # 1. Preferred: decode a single JSON object starting at the first '{'.
    #    raw_decode tolerates trailing prose after the object.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Bracket-match the outermost {...} (handles leading prose + nested braces).
    candidate = _bracket_match(text, start)
    if candidate is None:
        # Last resort: span from first '{' to last '}'.
        end = text.rfind("}")
        if end <= start:
            raise ValueError(f"Judge response is not JSON: {raw[:200]}")
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3. Safe repair: strip trailing commas, then retry.
    try:
        return json.loads(_strip_trailing_commas(candidate))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response is not valid JSON: {raw[:200]}") from exc


def _as_string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_result(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object")

    result = dict(RESULT_TEMPLATE)
    result["verification_type"] = VERIFICATION_TYPE

    verdict = str(data.get("verdict", "FAIL")).strip().upper()
    if verdict not in {"PASS", "FAIL"}:
        verdict = "FAIL"
    result["verdict"] = verdict

    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    result["score"] = max(0.0, min(1.0, score))

    result["missing_required_elements"] = _as_string_list(
        data.get("missing_required_elements"), "missing_required_elements"
    )
    result["incorrect_claims"] = _as_string_list(
        data.get("incorrect_claims"), "incorrect_claims"
    )
    result["unsupported_claims"] = _as_string_list(
        data.get("unsupported_claims"), "unsupported_claims"
    )
    result["disqualifying_conditions_triggered"] = _as_string_list(
        data.get("disqualifying_conditions_triggered"),
        "disqualifying_conditions_triggered",
    )
    result["explanation"] = str(data.get("explanation", "")).strip()

    # Enforce verdict consistency with hard failures
    if (
        result["disqualifying_conditions_triggered"]
        or result["missing_required_elements"]
    ):
        result["verdict"] = "FAIL"
    elif result["score"] < 0.75:
        result["verdict"] = "FAIL"
    elif (
        not result["disqualifying_conditions_triggered"]
        and not result["missing_required_elements"]
        and result["score"] >= 0.75
    ):
        result["verdict"] = "PASS"

    return result


def _call_judge(
    question: str,
    answer: str,
    specification: dict,
    *,
    model: str,
    retry: bool = False,
) -> dict:
    user_content = _build_user_message(question, answer, specification)
    if retry:
        user_content += (
            "\n\nReminder: respond with ONLY valid JSON matching the required schema."
        )

    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""
    parsed = _extract_json(raw)
    result = _normalize_result(parsed)
    result["model"] = model
    return result


def run(
    question: str,
    answer: str,
    specification: dict,
    *,
    model: str | None = None,
) -> dict:
    """
    Verify a specialist answer against a specification checklist.

    Args:
        question: Original user question.
        answer: Specialist-produced answer text.
        specification: Spec Generator output (intent, required_elements, etc.).

    Returns:
        Verification result dict with verdict, score, and issue lists.
    """
    question = question.strip()
    answer = answer.strip()
    if not question:
        raise ValueError("question cannot be empty")
    if not answer:
        raise ValueError("answer cannot be empty")

    _validate_specification(specification)
    model = model or DEFAULT_MODEL

    try:
        return _call_judge(question, answer, specification, model=model)
    except (ValueError, json.JSONDecodeError):
        try:
            return _call_judge(question, answer, specification, model=model, retry=True)
        except (ValueError, json.JSONDecodeError):
            # Unparseable even after a retry: degrade to a FAIL instead of
            # raising, so a single bad judge response cannot abort an eval run.
            return _parse_error_result(model)


def _parse_error_result(model: str) -> dict:
    """Normalized FAIL used when the judge response cannot be parsed."""
    result = _normalize_result(
        {
            "verdict": "FAIL",
            "score": 0.0,
            "explanation": "judge parse error",
        }
    )
    result["model"] = model
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an answer against a specification.")
    parser.add_argument("--question", "-q", help="Original user question")
    parser.add_argument("--answer", "-a", help="Specialist answer text")
    parser.add_argument("--spec", "-s", help="Path to specification JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA model id")
    args = parser.parse_args()

    question = args.question
    answer = args.answer
    spec_path = args.spec

    if not question:
        question = input("Question: ").strip()
    if not answer:
        answer = input("Answer: ").strip()
    if not spec_path:
        spec_path = input("Specification JSON path: ").strip()

    if not spec_path or not os.path.exists(spec_path):
        print("Error: specification file not found.", file=sys.stderr)
        sys.exit(1)

    with open(spec_path, encoding="utf-8") as f:
        specification = json.load(f)

    try:
        result = run(question, answer, specification, model=args.model)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
