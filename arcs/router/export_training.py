"""
Export ROUTER-attributed failures into DistilBERT training CSV rows.

Does not guess labels from ``route.domain`` — that may be the wrong prediction.
Requires ``expected_domain`` or ``correct_domain`` on each queue record.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from arcs import config

VALID_LABELS = frozenset({"CODING", "MEDICAL", "LEGAL", "GENERAL"})

DEFAULT_QUEUE = config.LOGS_DIR / "queues" / "router_queue.jsonl"
DEFAULT_CSV = config.ROUTER_DATA_DIR / "router_train.csv"


def _resolve_label(record: dict[str, Any]) -> str | None:
    """Return a valid domain label from expected_domain or correct_domain."""
    for key in ("expected_domain", "correct_domain"):
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        normalized = value.strip().upper()
        if normalized in VALID_LABELS:
            return normalized
    return None


def _read_queue(queue_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not queue_path.exists():
        raise FileNotFoundError(f"router queue not found: {queue_path}")

    with queue_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                print(
                    f"Warning: skipping non-object on line {line_number}",
                    file=sys.stderr,
                )
    return records


def _load_existing_pairs(csv_path: Path) -> set[tuple[str, str]]:
    """Load existing (text, label) pairs for deduplication."""
    pairs: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return pairs

    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            return pairs
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if text and label in VALID_LABELS:
                pairs.add((text, label))
    return pairs


def export_router_examples(
    queue_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    *,
    append: bool = True,
) -> int:
    """Export labeled router-queue records to a training CSV.

    Args:
        queue_path: Path to ``router_queue.jsonl``.
        output_csv: Destination CSV (``text,label``).
        append: When True, append new unique rows. When False, rewrite file
            with only newly exported rows (still deduped within the export).

    Returns:
        Number of new rows written.

    Raises:
        FileNotFoundError: If the queue file does not exist.
    """
    source = Path(queue_path) if queue_path is not None else DEFAULT_QUEUE
    destination = Path(output_csv) if output_csv is not None else DEFAULT_CSV

    records = _read_queue(source)
    existing = _load_existing_pairs(destination) if append else set()

    new_rows: list[tuple[str, str]] = []
    seen_this_batch: set[tuple[str, str]] = set()
    skipped_unlabeled = 0
    skipped_duplicate = 0

    for record in records:
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            print(
                f"Warning: skipping record with missing query "
                f"(query_id={record.get('query_id')!r})",
                file=sys.stderr,
            )
            continue

        text = query.strip()
        label = _resolve_label(record)
        if label is None:
            skipped_unlabeled += 1
            print(
                f"Warning: skipping unlabeled record "
                f"(query_id={record.get('query_id')!r}) — "
                "set expected_domain or correct_domain "
                "(do not guess from route.domain)",
                file=sys.stderr,
            )
            continue

        pair = (text, label)
        if pair in existing or pair in seen_this_batch:
            skipped_duplicate += 1
            continue

        seen_this_batch.add(pair)
        new_rows.append(pair)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if append and destination.exists():
        with destination.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            for text, label in new_rows:
                writer.writerow([text, label])
    else:
        with destination.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["text", "label"])
            for text, label in new_rows:
                writer.writerow([text, label])

    if skipped_unlabeled:
        print(
            f"Skipped {skipped_unlabeled} unlabeled row(s). "
            "Label them with scripts/label_router_failures.py",
            file=sys.stderr,
        )
    if skipped_duplicate:
        print(f"Skipped {skipped_duplicate} duplicate row(s).", file=sys.stderr)

    return len(new_rows)
