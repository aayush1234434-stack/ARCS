"""
Feedback Layer — collect user signals after a request completes.

This module runs after inference. It does not call LLMs, execute code, or
modify the pipeline state in place. It only collects explicit or interactive
feedback and prepares a post-inference record for logging and attribution.

Usage:
    from feedback import collect, apply

    signal = collect(explicit="NEGATIVE")
    record = apply(state, signal)
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import attribution

Signal = Literal["POSITIVE", "NEGATIVE"]
Source = Literal["explicit", "interactive", "retroactive"]

VALID_SIGNALS = frozenset({"POSITIVE", "NEGATIVE"})

LOG_DIR = Path("logs")
FEEDBACK_LOG_FILE = LOG_DIR / "feedback.jsonl"
REQUEST_LOG_FILE = LOG_DIR / "requests.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_signal(signal: str) -> Signal:
    normalized = signal.strip().upper()
    if normalized not in VALID_SIGNALS:
        raise ValueError(f"feedback signal must be POSITIVE or NEGATIVE, got {signal!r}")
    return normalized  # type: ignore[return-value]


def _derive_status(state: dict[str, Any]) -> str:
    verdict = state.get("verification", {}).get("verdict")
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def _build_feedback(signal: Signal, source: Source) -> dict[str, Any]:
    return {
        "user_feedback": signal,
        "user_signal": signal,
        "source": source,
        "collected_at": _now_iso(),
    }


def collect_explicit(signal: str) -> dict[str, Any]:
    """Record an explicit user signal from CLI or API input."""
    return _build_feedback(_normalize_signal(signal), "explicit")


def collect_interactive() -> dict[str, Any] | None:
    """Prompt the user for feedback on stderr; return None if skipped."""
    print("Was this answer helpful? [y]es / [n]o / [s]kip: ", file=sys.stderr, end="", flush=True)
    choice = input().strip().lower()
    if choice in {"y", "yes"}:
        return _build_feedback("POSITIVE", "interactive")
    if choice in {"n", "no"}:
        return _build_feedback("NEGATIVE", "interactive")
    return None


def collect(
    *,
    explicit: str | None = None,
    interactive: bool = False,
) -> dict[str, Any] | None:
    """Collect a user feedback signal after the pipeline has finished.

    Args:
        explicit: Pre-supplied signal (``POSITIVE`` or ``NEGATIVE``).
        interactive: When True and ``explicit`` is omitted, prompt the user.

    Returns:
        Feedback dict, or ``None`` when no signal was collected.
    """
    if explicit is not None:
        return collect_explicit(explicit)
    if interactive:
        return collect_interactive()
    return None


def apply(state: dict, feedback: dict[str, Any] | None) -> dict[str, Any]:
    """Merge pipeline state with optional feedback and attribution.

    Does not modify ``state`` in place.

    Args:
        state: Pipeline execution state from ``run_pipeline``.
        feedback: Output from :func:`collect`, or ``None``.

    Returns:
        Log-ready record with ``status``, optional ``feedback``, and
        ``attribution`` when feedback is ``NEGATIVE``.
    """
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state).__name__}")
    if feedback is not None and not isinstance(feedback, dict):
        raise TypeError(f"feedback must be a dict or None, got {type(feedback).__name__}")

    record = deepcopy(state)
    record["status"] = _derive_status(state)

    if feedback is None:
        return record

    signal = feedback.get("user_feedback")
    if signal is not None:
        record["user_feedback"] = _normalize_signal(str(signal))
        record["user_signal"] = record["user_feedback"]

    record["feedback"] = dict(feedback)

    if record["user_feedback"] == "NEGATIVE":
        record["attribution"] = attribution.attribute(record)

    return record


def log_event(record: dict[str, Any], log_file: Path = FEEDBACK_LOG_FILE) -> str:
    """Append a feedback event to ``logs/feedback.jsonl``."""
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}")

    event = {
        "timestamp": _now_iso(),
        "query_id": record.get("query_id") or record.get("metadata", {}).get("query_id"),
        "user_feedback": record.get("user_feedback"),
        "user_signal": record.get("user_signal"),
        "feedback": record.get("feedback"),
        "attribution": record.get("attribution"),
        "query": record.get("query"),
        "status": record.get("status"),
    }

    try:
        line = json.dumps(event, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise TypeError(f"record is not JSON-serializable: {exc}") from exc

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        raise OSError(f"failed to write feedback event to {log_file}: {exc}") from exc

    return str(log_file)


def apply_to_logged_request(
    query_id: str,
    signal: str,
    *,
    request_log: Path = REQUEST_LOG_FILE,
) -> dict[str, Any]:
    """Load a prior request, attach feedback, attribute, and log the event."""
    record = attribution.load_record(query_id, request_log)
    feedback_signal = collect_explicit(signal)
    feedback_signal["source"] = "retroactive"
    enriched = apply(record, feedback_signal)
    log_event(enriched)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach user feedback to a previously logged ARCS request.",
    )
    parser.add_argument("--query-id", required=True, help="query_id from logs/requests.jsonl")
    parser.add_argument(
        "--signal",
        choices=sorted(VALID_SIGNALS),
        required=True,
        help="User feedback signal",
    )
    parser.add_argument(
        "--request-log",
        default=str(REQUEST_LOG_FILE),
        help="Path to the request JSONL log",
    )
    args = parser.parse_args()

    try:
        record = apply_to_logged_request(
            args.query_id,
            args.signal,
            request_log=Path(args.request_log),
        )
    except (FileNotFoundError, LookupError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(record.get("attribution") or record.get("feedback"), indent=2))


if __name__ == "__main__":
    main()
