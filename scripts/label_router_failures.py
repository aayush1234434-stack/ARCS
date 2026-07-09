"""CLI: set correct_domain on router-queue rows missing labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.router.export_training import VALID_LABELS

DEFAULT_QUEUE = config.LOGS_DIR / "queues" / "router_queue.jsonl"


def _load_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"router queue not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
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
    return records


def _write_queue(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _has_label(record: dict[str, Any]) -> bool:
    for key in ("expected_domain", "correct_domain"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().upper() in VALID_LABELS:
            return True
    return False


def _preview(text: str, limit: int = 100) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def label_by_query_id(
    records: list[dict[str, Any]],
    query_id: str,
    correct_domain: str,
) -> int:
    """Set correct_domain on matching query_id. Returns number of updates."""
    domain = correct_domain.strip().upper()
    if domain not in VALID_LABELS:
        raise ValueError(f"correct_domain must be one of {sorted(VALID_LABELS)}")

    updated = 0
    for record in records:
        if record.get("query_id") == query_id:
            record["correct_domain"] = domain
            updated += 1
    return updated


def label_interactive(records: list[dict[str, Any]]) -> int:
    """Prompt for labels on unlabeled rows. Returns number of updates."""
    unlabeled = [r for r in records if not _has_label(r)]
    if not unlabeled:
        print("No unlabeled router-queue rows.", file=sys.stderr)
        return 0

    print(
        f"{len(unlabeled)} unlabeled row(s). "
        f"Enter domain ({'/'.join(sorted(VALID_LABELS))}) or [s]kip / [q]uit.",
        file=sys.stderr,
    )
    updated = 0
    for index, record in enumerate(unlabeled, start=1):
        query = str(record.get("query", ""))
        routed = (record.get("route") or {}).get("domain", "?")
        conf = (record.get("route") or {}).get("confidence", "?")
        print(
            f"\n[{index}/{len(unlabeled)}] query_id={record.get('query_id')}",
            file=sys.stderr,
        )
        print(f"  query: {_preview(query)}", file=sys.stderr)
        print(f"  routed_as: {routed} (confidence={conf})", file=sys.stderr)
        print("  correct domain: ", file=sys.stderr, end="", flush=True)
        choice = input().strip()
        if choice.lower() in {"q", "quit"}:
            break
        if choice.lower() in {"s", "skip", ""}:
            continue
        domain = choice.upper()
        if domain not in VALID_LABELS:
            print(f"  Invalid domain {choice!r} — skipped.", file=sys.stderr)
            continue
        record["correct_domain"] = domain
        updated += 1
        print(f"  Set correct_domain={domain}", file=sys.stderr)
    return updated


def list_unlabeled(records: list[dict[str, Any]]) -> None:
    unlabeled = [r for r in records if not _has_label(r)]
    if not unlabeled:
        print("No unlabeled router-queue rows.")
        return
    for record in unlabeled:
        print(
            f"{record.get('query_id')}\t"
            f"routed={(record.get('route') or {}).get('domain')}\t"
            f"{_preview(str(record.get('query', '')), 60)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set correct_domain on router-queue failures so they can be "
            "exported into router_train.csv."
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help=f"Path to router_queue.jsonl (default: {DEFAULT_QUEUE})",
    )
    parser.add_argument(
        "--query-id",
        help="Label a single record by query_id",
    )
    parser.add_argument(
        "--correct-domain",
        choices=sorted(VALID_LABELS),
        help="Domain label to set (required with --query-id)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively label all unlabeled rows",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List unlabeled query_ids and exit",
    )
    args = parser.parse_args()

    try:
        records = _load_queue(args.queue)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Run: python scripts/extract_queues.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list:
        list_unlabeled(records)
        return

    if args.query_id is not None:
        if not args.correct_domain:
            print("Error: --correct-domain is required with --query-id", file=sys.stderr)
            sys.exit(1)
        try:
            updated = label_by_query_id(records, args.query_id, args.correct_domain)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        if updated == 0:
            print(f"Error: no record with query_id={args.query_id!r}", file=sys.stderr)
            sys.exit(1)
        _write_queue(args.queue, records)
        print(f"Updated {updated} record(s) → correct_domain={args.correct_domain.upper()}")
        return

    if args.interactive:
        updated = label_interactive(records)
        if updated:
            _write_queue(args.queue, records)
            print(f"Saved {updated} label(s) to {args.queue}", file=sys.stderr)
        else:
            print("No changes saved.", file=sys.stderr)
        return

    parser.print_help()
    print(
        "\nExamples:\n"
        "  python scripts/label_router_failures.py --list\n"
        "  python scripts/label_router_failures.py --query-id <uuid> --correct-domain MEDICAL\n"
        "  python scripts/label_router_failures.py --interactive",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
