"""
Append pipeline state records to a JSON Lines log file.

This module serializes whatever dictionary it receives and enriches each
record with top-level status, response, models, timing, and metadata before
writing. It does not import or depend on other ARCS components.

Usage:
    from logger import log

    log_path = log(state)
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "requests.jsonl"


def _derive_status(state: dict[str, Any]) -> str:
    """Map verification verdict to a top-level pipeline status."""
    if state.get("error"):
        return "ERROR"

    verdict = state.get("verification", {}).get("verdict")
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def _derive_response(state: dict[str, Any]) -> str:
    """Extract the final user-facing answer from specialist output."""
    return str(state.get("specialist", {}).get("answer", ""))


def _collect_models(state: dict[str, Any]) -> dict[str, str]:
    """Gather model identifiers from nested component outputs."""
    models: dict[str, str] = {}

    router_model = state.get("route", {}).get("model")
    if router_model:
        models["router"] = str(router_model)

    specialist = state.get("specialist", {})
    generator_model = specialist.get("generator_model") or specialist.get("specialist")
    if generator_model:
        models["generator"] = str(generator_model)
        # Keep legacy key for older log consumers.
        models["specialist"] = str(generator_model)

    spec_model = state.get("specification", {}).get("model")
    if spec_model:
        models["spec_generator"] = str(spec_model)

    test_model = state.get("tooling", {}).get("test_generator_model")
    if test_model:
        models["test_generator"] = str(test_model)

    verification = state.get("verification", {})
    verifier_model = verification.get("model")
    if verifier_model:
        models["verifier"] = str(verifier_model)
    elif verification.get("verification_type") == "SANDBOX":
        models["verifier"] = "sandbox"

    return models


def _collect_metadata(
    state: dict[str, Any],
    *,
    query_id: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build operational metadata for downstream attribution and analytics."""
    route = state.get("route", {})
    specialist = state.get("specialist", {})
    verification = state.get("verification", {})

    pipeline = state.get("pipeline", {})
    tooling = state.get("tooling", {})

    return {
        "query_id": query_id,
        "timestamp": timestamp,
        "domain": route.get("domain"),
        "pipeline_id": pipeline.get("pipeline_id") or specialist.get("pipeline_id"),
        "specialist_domain": specialist.get("domain"),
        "use_fallback": route.get("use_fallback"),
        "router_confidence": route.get("confidence"),
        "generator_model": (
            pipeline.get("generator_model")
            or specialist.get("generator_model")
            or specialist.get("specialist")
        ),
        "verifier_type": verification.get("verification_type"),
        "verdict": verification.get("verdict"),
        "score": verification.get("score"),
        "specialist_uncertainty": specialist.get("specialist_uncertainty"),
        "sandbox_rounds_used": tooling.get("rounds_used"),
        "delivery_warning": tooling.get("delivery_warning"),
        "user_feedback": state.get("user_feedback"),
        "user_signal": state.get("user_signal"),
        "feedback": state.get("feedback"),
        "attribution": state.get("attribution"),
    }


def _build_entry(state: dict[str, Any]) -> dict[str, Any]:
    """Enrich a pipeline state dict into a log record without mutating input."""
    query_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = deepcopy(state)
    timing = payload.pop("timing", {})
    if not isinstance(timing, dict):
        timing = {}

    return {
        "query_id": query_id,
        "timestamp": timestamp,
        "status": _derive_status(state),
        "response": _derive_response(state),
        "models": _collect_models(state),
        "timing": timing,
        "metadata": _collect_metadata(state, query_id=query_id, timestamp=timestamp),
        **payload,
    }


def log(state: dict) -> str:
    """Append one enriched state record to ``logs/requests.jsonl``.

    Args:
        state: Pipeline state (or any JSON-serializable mapping) to persist.

    Returns:
        Path to the log file as a string.

    Raises:
        TypeError: If ``state`` is not a dict or contains non-serializable values.
        OSError: If the log directory or file cannot be written.
    """
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state).__name__}")

    entry = _build_entry(state)

    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise TypeError(f"state is not JSON-serializable: {exc}") from exc

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        raise OSError(f"failed to write log entry to {LOG_FILE}: {exc}") from exc

    return str(LOG_FILE)
