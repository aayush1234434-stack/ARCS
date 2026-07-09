"""CLI: validate Phase 2 held-out eval queries JSONL.

Checks schema, domain enum, unique ids, and prints per-domain counts.
Does not call the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config

VALID_DOMAINS = frozenset({"CODING", "MEDICAL", "LEGAL", "GENERAL"})
DEFAULT_INPUT = config.DATA_DIR / "eval_queries.jsonl"
REQUIRED_FIELDS = ("id", "query", "expected_domain")


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"file not found: {path}"]

    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: expected a JSON object")
                continue
            rows.append(row)
            # Attach line number for later error messages.
            row["_line"] = line_number
    return rows, errors


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for row in rows:
        line = row.get("_line", "?")

        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"line {line}: missing required field {field!r}")

        row_id = row.get("id")
        if row_id is not None:
            if not isinstance(row_id, str) or not row_id.strip():
                errors.append(f"line {line}: id must be a non-empty string")
            elif row_id in seen_ids:
                errors.append(f"line {line}: duplicate id {row_id!r}")
            else:
                seen_ids.add(row_id)

        query = row.get("query")
        if query is not None and (not isinstance(query, str) or not query.strip()):
            errors.append(f"line {line}: query must be a non-empty string")

        domain = row.get("expected_domain")
        if domain is not None:
            if not isinstance(domain, str) or domain.strip().upper() not in VALID_DOMAINS:
                errors.append(
                    f"line {line}: expected_domain must be one of "
                    f"{sorted(VALID_DOMAINS)}, got {domain!r}"
                )

        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            errors.append(f"line {line}: notes must be a string when present")

        tags = row.get("tags")
        if tags is not None:
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                errors.append(f"line {line}: tags must be a list of strings when present")

        unknown = set(row) - {
            "id",
            "query",
            "expected_domain",
            "notes",
            "tags",
            "_line",
        }
        if unknown:
            errors.append(
                f"line {line}: unexpected field(s): {', '.join(sorted(unknown))}"
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate data/eval_queries.jsonl (Phase 2 held-out eval set). "
            "Checks schema and domain enum; prints per-domain counts."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Eval JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=40,
        help="Minimum valid rows required (default: 40)",
    )
    args = parser.parse_args()

    rows, load_errors = _load_rows(args.input)
    errors = list(load_errors)
    errors.extend(validate_rows(rows))

    counts = Counter()
    for row in rows:
        domain = row.get("expected_domain")
        if isinstance(domain, str) and domain.strip().upper() in VALID_DOMAINS:
            counts[domain.strip().upper()] += 1

    print(f"file: {args.input}")
    print(f"rows: {len(rows)}")
    print("per-domain:")
    for domain in sorted(VALID_DOMAINS):
        print(f"  {domain}: {counts.get(domain, 0)}")

    if len(rows) < args.min_rows:
        errors.append(
            f"need at least {args.min_rows} rows, found {len(rows)}"
        )

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print("validation: OK")


if __name__ == "__main__":
    main()
