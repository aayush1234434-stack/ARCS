"""Prepare RQ1 router training datasets from the synthetic feedback corpus.

Builds two augmented training sets so RQ1 can compare:
  - Run A: augment the base router train set with EVERY corpus negative
           (all rows carrying a correct_domain label).
  - Run B: augment only with corpus rows attributed to the ROUTER component.

The held-out test set (data/router/router_test.csv) is NEVER read or written
here — it must stay untouched for a fair RQ1 evaluation.

Inputs:
    data/rq1/feedback_corpus.jsonl   (see scripts/bootstrap_rq1_corpus.py)
    data/router/router_train.csv     (base train, copied read-only)

Outputs (under data/router/rq1/):
    base_train.csv        exact copy of router_train.csv
    run_a_augment.csv     text,label from all corpus rows with correct_domain
    run_b_augment.csv     text,label from ROUTER-attributed corpus rows only
    run_a_train.csv       base_train + run_a_augment (deduped)
    run_b_train.csv       base_train + run_b_augment (deduped)

Usage:
    python scripts/rq1_prepare_datasets.py --dry-run
    python scripts/rq1_prepare_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config

DEFAULT_CORPUS = config.DATA_DIR / "rq1" / "feedback_corpus.jsonl"
BASE_TRAIN_CSV = config.ROUTER_DATA_DIR / "router_train.csv"
OUTPUT_DIR = config.ROUTER_DATA_DIR / "rq1"

MIN_RUN_B_AUGMENT = 5

Pair = tuple[str, str]


def _norm_key(text: str, label: str) -> Pair:
    """Normalized (text, label) key for dedupe — case/whitespace-insensitive."""
    return " ".join(text.strip().lower().split()), label.strip().upper()


def _load_base(path: Path) -> list[Pair]:
    """Read base train rows as (text, label), preserving original order/casing."""
    rows: list[Pair] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows.append((text, label))
    return rows


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _augment_pairs(records: list[dict[str, Any]], *, router_only: bool) -> list[Pair]:
    """Extract deduped (query, correct_domain) pairs from corpus records."""
    pairs: list[Pair] = []
    seen: set[Pair] = set()
    for rec in records:
        if router_only and (rec.get("attribution") or {}).get("component") != "ROUTER":
            continue
        text = str(rec.get("query") or "").strip()
        label = str(rec.get("correct_domain") or "").strip()
        if not text or not label:
            continue
        key = _norm_key(text, label)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((text, label))
    return pairs


def _merge(base: list[Pair], augment: list[Pair]) -> tuple[list[Pair], int]:
    """Merge base + augment, deduping (text,label). Returns (merged, overlap).

    ``overlap`` is the number of augment pairs that were already present in the
    base set (so they did not increase the total).
    """
    seen: set[Pair] = set()
    merged: list[Pair] = []
    for text, label in base:
        key = _norm_key(text, label)
        if key not in seen:
            seen.add(key)
            merged.append((text, label))

    base_keys = set(seen)
    overlap = 0
    for text, label in augment:
        key = _norm_key(text, label)
        if key in base_keys:
            overlap += 1
        if key not in seen:
            seen.add(key)
            merged.append((text, label))
    return merged, overlap


def _write_csv(path: Path, rows: list[Pair]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RQ1 augmented router training datasets.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Feedback corpus JSONL (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print counts without writing any files.",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)
    if not BASE_TRAIN_CSV.exists():
        print(f"Error: base train not found: {BASE_TRAIN_CSV}", file=sys.stderr)
        sys.exit(1)

    base = _load_base(BASE_TRAIN_CSV)
    records = _load_corpus(args.corpus)

    run_a_aug = _augment_pairs(records, router_only=False)
    run_b_aug = _augment_pairs(records, router_only=True)

    run_a_train, run_a_overlap = _merge(base, run_a_aug)
    run_b_train, _ = _merge(base, run_b_aug)

    print("RQ1 dataset preparation")
    print(f"  base:        {len(base)}")
    print(f"  run_a_aug:   {len(run_a_aug)}")
    print(f"  run_b_aug:   {len(run_b_aug)}")
    print(f"  run_a_total: {len(run_a_train)}")
    print(f"  run_b_total: {len(run_b_train)}")
    print(f"  overlap:     {run_a_overlap}  (run_a_aug pairs already in base)")

    # ── Validation ──
    errors: list[str] = []
    if len(run_a_aug) < len(run_b_aug):
        errors.append(
            f"run_a_aug ({len(run_a_aug)}) must be >= run_b_aug ({len(run_b_aug)})"
        )
    if len(run_b_aug) < MIN_RUN_B_AUGMENT:
        errors.append(
            f"run_b_aug ({len(run_b_aug)}) must be >= {MIN_RUN_B_AUGMENT}; "
            "not enough ROUTER-attributed feedback to form Run B"
        )
    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE_TRAIN_CSV, OUTPUT_DIR / "base_train.csv")
    _write_csv(OUTPUT_DIR / "run_a_augment.csv", run_a_aug)
    _write_csv(OUTPUT_DIR / "run_b_augment.csv", run_b_aug)
    _write_csv(OUTPUT_DIR / "run_a_train.csv", run_a_train)
    _write_csv(OUTPUT_DIR / "run_b_train.csv", run_b_train)

    print(f"\nWrote 5 file(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
