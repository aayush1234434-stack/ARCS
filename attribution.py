"""
Attribution Engine — assign blame to one ARCS component after a request completes.

This module is deterministic and stateless. It does not call LLMs, execute code,
or modify the input state. It only inspects the execution record and applies
the fixed rule set in order.

Usage:
    from attribution import attribute

    result = attribute(state)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Component = Literal["SPECIALIST", "VERIFIER", "ROUTER", "AMBIGUOUS"]

ATTRIBUTION_KEYS = frozenset({"component", "reason", "rule"})


def _section(state: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mapping from state, or an empty dict if absent/invalid."""
    value = state.get(key, {})
    return value if isinstance(value, dict) else {}


def _user_feedback(state: dict[str, Any]) -> str | None:
    """Return normalized user feedback when explicitly present."""
    feedback = state.get("user_feedback")
    if feedback is None:
        return None
    if not isinstance(feedback, str):
        raise TypeError("user_feedback must be a string when provided")
    return feedback.strip().upper()


def _verification_score(verification: dict[str, Any]) -> float | None:
    """Parse verification score when present and numeric."""
    score = verification.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError) as exc:
        raise TypeError("verification score must be numeric when provided") from exc


def _router_confidence(route: dict[str, Any]) -> float | None:
    """Parse router confidence when present and numeric."""
    confidence = route.get("confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError) as exc:
        raise TypeError("route confidence must be numeric when provided") from exc


def _specialist_uncertainty(state: dict[str, Any]) -> bool | None:
    """Return specialist uncertainty flag from specialist output or top-level state."""
    specialist = _section(state, "specialist")
    if "specialist_uncertainty" in specialist:
        value = specialist["specialist_uncertainty"]
    elif "specialist_uncertainty" in state:
        value = state["specialist_uncertainty"]
    else:
        return None

    if not isinstance(value, bool):
        raise TypeError("specialist_uncertainty must be a boolean when provided")
    return value


def _result(component: Component, reason: str, rule: int) -> dict[str, Any]:
    """Build a normalized attribution result."""
    return {
        "component": component,
        "reason": reason,
        "rule": rule,
    }


def _rule_1_sandbox_failure(verification: dict[str, Any]) -> dict[str, Any] | None:
    if verification.get("verification_type") != "SANDBOX":
        return None
    if verification.get("verdict") != "FAIL":
        return None
    return _result(
        "SPECIALIST",
        "Sandbox execution failed.",
        1,
    )


def _rule_2_verifier_overconfidence(
    verification: dict[str, Any],
    user_feedback: str | None,
) -> dict[str, Any] | None:
    if user_feedback != "NEGATIVE":
        return None

    score = _verification_score(verification)
    if score is None or score <= 0.80:
        return None

    return _result(
        "VERIFIER",
        "Verifier approved an incorrect answer with high confidence.",
        2,
    )


def _rule_3_router_low_confidence(
    route: dict[str, Any],
    user_feedback: str | None,
) -> dict[str, Any] | None:
    if user_feedback != "NEGATIVE":
        return None

    confidence = _router_confidence(route)
    if confidence is None or confidence >= 0.60:
        return None

    return _result(
        "ROUTER",
        "Router dispatched with low confidence.",
        3,
    )


def _rule_4_specialist_uncertainty(
    state: dict[str, Any],
    user_feedback: str | None,
) -> dict[str, Any] | None:
    if user_feedback != "NEGATIVE":
        return None

    uncertainty = _specialist_uncertainty(state)
    if uncertainty is not True:
        return None

    return _result(
        "SPECIALIST",
        "Specialist already indicated uncertainty.",
        4,
    )


def attribute(state: dict) -> dict:
    """Assign responsibility for a failure using the fixed ARCS rule set.

    Rules are evaluated in order and the first matching rule wins.

    Args:
        state: Complete execution state from the pipeline or a logged record.

    Returns:
        Attribution result with keys ``component``, ``reason``, and ``rule``.

    Raises:
        TypeError: If ``state`` is not a dict or required fields have invalid types.
    """
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state).__name__}")

    verification = _section(state, "verification")
    route = _section(state, "route")
    feedback = _user_feedback(state)

    for rule_fn, args in (
        (_rule_1_sandbox_failure, (verification,)),
        (_rule_2_verifier_overconfidence, (verification, feedback)),
        (_rule_3_router_low_confidence, (route, feedback)),
        (_rule_4_specialist_uncertainty, (state, feedback)),
    ):
        result = rule_fn(*args)
        if result is not None:
            return result

    return _result(
        "AMBIGUOUS",
        "Unable to confidently attribute failure.",
        5,
    )


def load_record(query_id: str, log_file: str | Path = "logs/requests.jsonl") -> dict[str, Any]:
    """Load a single log record by ``query_id`` from a JSON Lines file.

    Args:
        query_id: UUID of the request to look up.
        log_file: Path to the JSONL log file.

    Returns:
        The matching log record.

    Raises:
        FileNotFoundError: If the log file does not exist.
        LookupError: If no record matches ``query_id``.
        ValueError: If a log line is not valid JSON or not an object.
    """
    path = Path(log_file)
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"log line {line_number} in {path} is not a JSON object")
            if record.get("query_id") == query_id:
                return record

    raise LookupError(f"no log record found for query_id={query_id!r} in {path}")


def main() -> None:
    """Attribute a previously logged request after user feedback is known."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run attribution on a logged ARCS request.",
    )
    parser.add_argument(
        "--query-id",
        required=True,
        help="query_id from logs/requests.jsonl",
    )
    parser.add_argument(
        "--feedback",
        choices=("POSITIVE", "NEGATIVE"),
        required=True,
        help="User feedback signal",
    )
    parser.add_argument(
        "--log-file",
        default="logs/requests.jsonl",
        help="Path to the JSONL log file",
    )
    args = parser.parse_args()

    try:
        record = load_record(args.query_id, args.log_file)
        record["user_feedback"] = args.feedback
        result = attribute(record)
    except (FileNotFoundError, LookupError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
