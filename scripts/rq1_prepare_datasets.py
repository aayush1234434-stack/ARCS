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
REAL_CORPUS = config.DATA_DIR / "rq1" / "feedback_corpus_real.jsonl"
BASE_TRAIN_CSV = config.ROUTER_DATA_DIR / "router_train.csv"
OUTPUT_DIR = config.ROUTER_DATA_DIR / "rq1"

MIN_RUN_B_AUGMENT = 5
RQ1_V2_MIN_NEGATIVES = 40
RQ1_V2_MIN_ROUTER = 15

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


def _resolve_corpus(value: str) -> tuple[Path, str]:
    if value == "real":
        return REAL_CORPUS, "real"
    path = Path(value)
    kind = "real" if path.resolve() == REAL_CORPUS.resolve() or "real" in path.name else "bootstrap"
    return path, kind


def _corpus_attribution_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        component = (rec.get("attribution") or {}).get("component") or "UNKNOWN"
        counts[component] = counts.get(component, 0) + 1
    return counts


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
        default=str(DEFAULT_CORPUS),
        help=(
            f'Feedback corpus JSONL, or "real" for {REAL_CORPUS} '
            f"(default: {DEFAULT_CORPUS})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print counts without writing any files.",
    )
    args = parser.parse_args()
    corpus_path, corpus_kind = _resolve_corpus(args.corpus)

    if not corpus_path.exists():
        print(f"Error: corpus not found: {corpus_path}", file=sys.stderr)
        if corpus_kind == "real":
            print(
                "Build with: python scripts/bootstrap_rq1_corpus.py --real-only",
                file=sys.stderr,
            )
        sys.exit(1)
    if not BASE_TRAIN_CSV.exists():
        print(f"Error: base train not found: {BASE_TRAIN_CSV}", file=sys.stderr)
        sys.exit(1)

    base = _load_base(BASE_TRAIN_CSV)
    records = _load_corpus(corpus_path)

    run_a_aug = _augment_pairs(records, router_only=False)
    run_b_aug = _augment_pairs(records, router_only=True)

    run_a_train, run_a_overlap = _merge(base, run_a_aug)
    run_b_train, run_b_overlap = _merge(base, run_b_aug)

    attr_counts = _corpus_attribution_counts(records)
    run_a_only = len(run_a_aug) - len(run_b_aug)

    print("RQ1 dataset preparation")
    print(f"  corpus:      {corpus_path} ({corpus_kind})")
    print(f"  base:        {len(base)}")
    print(f"  corpus rows: {len(records)}")
    print("  Run A augment (all negatives with correct_domain):")
    print(f"    rows:      {len(run_a_aug)}")
    print("  Run B augment (ROUTER-attributed only):")
    print(f"    rows:      {len(run_b_aug)}")
    print(f"  Run A − Run B: {run_a_only} row(s) in Run A only (non-ROUTER)")
    print(f"  run_a_total: {len(run_a_train)}")
    print(f"  run_b_total: {len(run_b_train)}")
    print(f"  overlap:     run_a={run_a_overlap}  run_b={run_b_overlap}  (aug pairs already in base)")
    if attr_counts:
        print("  corpus attribution:")
        for component in ("ROUTER", "SPECIALIST", "VERIFIER", "AMBIGUOUS", "UNKNOWN"):
            if component in attr_counts:
                print(f"    {component:11s} {attr_counts[component]}")

    # ── Validation ──
    errors: list[str] = []
    if len(run_a_aug) < len(run_b_aug):
        errors.append(
            f"run_a_aug ({len(run_a_aug)}) must be >= run_b_aug ({len(run_b_aug)})"
        )

    if corpus_kind == "real":
        router_in_corpus = attr_counts.get("ROUTER", 0)
        if len(records) < RQ1_V2_MIN_NEGATIVES:
            errors.append(
                f"total negatives {len(records)} < {RQ1_V2_MIN_NEGATIVES} "
                f"(need more live 👎 feedback in logs/requests.jsonl)"
            )
        if router_in_corpus < RQ1_V2_MIN_ROUTER:
            errors.append(
                f"ROUTER-attributed negatives {router_in_corpus} < {RQ1_V2_MIN_ROUTER} "
                f"(thumb down on misrouted queries and pick correct_domain in the demo)"
            )
        if len(run_b_aug) < RQ1_V2_MIN_ROUTER:
            errors.append(
                f"run_b_aug ({len(run_b_aug)}) < {RQ1_V2_MIN_ROUTER} "
                f"(ROUTER rows need correct_domain labels for Run B training)"
            )
    elif len(run_b_aug) < MIN_RUN_B_AUGMENT:
        errors.append(
            f"run_b_aug ({len(run_b_aug)}) must be >= {MIN_RUN_B_AUGMENT}; "
            "not enough ROUTER-attributed feedback to form Run B"
        )

    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if corpus_kind == "real":
            print(
                "\nCheck readiness: python scripts/feedback_stats.py --requests-only",
                file=sys.stderr,
            )
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
