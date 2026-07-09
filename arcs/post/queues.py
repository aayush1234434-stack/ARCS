"""
Queue extraction — sort NEGATIVE feedback logs into per-component piles.

This module is a filter and groupby only. It does not call LLMs, retrain
models, or change attribution decisions. Attribution already decided blame;
this script organizes those decisions for downstream repair (router retrain,
DSPy, etc.).

Usage:
    from arcs.post.queues import extract_queues

    queues = extract_queues()
    print({k: len(v) for k, v in queues.items()})
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from arcs import config

COMPONENTS = ("ROUTER", "SPECIALIST", "VERIFIER", "AMBIGUOUS")

_QUEUE_FILENAMES = {
    "ROUTER": "router_queue.jsonl",
    "SPECIALIST": "specialist_queue.jsonl",
    "VERIFIER": "verifier_queue.jsonl",
    "AMBIGUOUS": "ambiguous_queue.jsonl",
}


def _user_feedback(record: dict[str, Any]) -> str | None:
    """Return normalized feedback from top-level or metadata."""
    feedback = record.get("user_feedback")
    if feedback is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            feedback = metadata.get("user_feedback")
    if feedback is None:
        return None
    if not isinstance(feedback, str):
        return None
    return feedback.strip().upper()


def _attribution_component(record: dict[str, Any]) -> str | None:
    """Return attribution.component when present and valid."""
    attribution = record.get("attribution")
    if not isinstance(attribution, dict):
        return None
    component = attribution.get("component")
    if not isinstance(component, str):
        return None
    normalized = component.strip().upper()
    if normalized not in COMPONENTS:
        return None
    return normalized


def _empty_queues() -> dict[str, list[dict[str, Any]]]:
    return {component: [] for component in COMPONENTS}


def _read_records(input_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records, skipping blank lines and invalid JSON."""
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number} "
                    f"of {input_path}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                print(
                    f"Warning: skipping non-object JSON on line {line_number} "
                    f"of {input_path}",
                    file=sys.stderr,
                )
                continue
            records.append(record)
    return records


def _write_queues(
    queues: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for component, records in queues.items():
        path = output_dir / _QUEUE_FILENAMES[component]
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_queues(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Filter NEGATIVE feedback logs into per-component queues.

    Args:
        input_path: Path to ``requests.jsonl``. Defaults to ``logs/requests.jsonl``.
        output_dir: Directory for queue files. Defaults to ``logs/queues/``.
        dry_run: When True, bucket in memory but do not write files.

    Returns:
        Mapping of component name → list of matching log records.

    Raises:
        FileNotFoundError: If the input log file does not exist.
    """
    source = Path(input_path) if input_path is not None else config.LOGS_DIR / "requests.jsonl"
    destination = (
        Path(output_dir) if output_dir is not None else config.LOGS_DIR / "queues"
    )

    if not source.exists():
        raise FileNotFoundError(
            f"request log not found: {source}\n"
            "Run the pipeline with --feedback NEGATIVE first, or pass --input."
        )

    queues = _empty_queues()
    for record in _read_records(source):
        if _user_feedback(record) != "NEGATIVE":
            continue
        component = _attribution_component(record)
        if component is None:
            print(
                "Warning: skipping NEGATIVE record with missing/invalid "
                f"attribution (query_id={record.get('query_id')!r})",
                file=sys.stderr,
            )
            continue
        queues[component].append(record)

    if not dry_run:
        _write_queues(queues, destination)

    return queues


def queue_counts(queues: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Return record counts per component, in COMPONENTS order."""
    return {component: len(queues.get(component, [])) for component in COMPONENTS}


def format_summary(counts: dict[str, int]) -> str:
    """Format counts as ``ROUTER: 3, SPECIALIST: 1, ...``."""
    return ", ".join(f"{component}: {counts.get(component, 0)}" for component in COMPONENTS)
