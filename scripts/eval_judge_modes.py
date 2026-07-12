"""Dry-run comparison of strict vs relaxed judge verdict policy on cached experiments.

Reads saved ``experiment.json`` rows (no API calls). Re-applies
``arcs.verification.judge.apply_verdict_policy`` in strict mode (default,
``JUDGE_STRICT=1``) and relaxed mode (``JUDGE_STRICT=0``: score >= 0.75,
at most one missing required element, no incorrect claims).

Usage:
    python scripts/eval_judge_modes.py --compare \\
        artifacts/experiments/2026-07-11T13-45-31_post-fix-v2-merged
    python scripts/eval_judge_modes.py --compare <path> --fail-only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.eval.compare import pass_stats
from arcs.eval.experiments import load_experiment
from arcs.verification.judge import apply_verdict_policy


def _rows_from_experiment(data: dict[str, Any]) -> list[dict[str, Any]]:
    meta = data.get("meta") or {}
    if isinstance(meta, dict) and isinstance(meta.get("rows"), list):
        return meta["rows"]
    rows = data.get("rows")
    return rows if isinstance(rows, list) else []


def _row_verdict(row: dict[str, Any]) -> str | None:
    verification = row.get("verification") or {}
    if isinstance(verification, dict) and verification.get("verdict"):
        return str(verification["verdict"]).upper()
    verdict = row.get("verdict")
    return str(verdict).upper() if verdict else None


def _row_domain(row: dict[str, Any]) -> str:
    for key in ("pipeline_id", "expected_domain", "domain"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return "UNKNOWN"


def _score_bucket(score: float) -> str:
    if score == 0.0:
        return "0.0"
    if 0.5 <= score <= 0.6:
        return "0.5-0.6"
    if score >= 0.7:
        return "0.7+"
    return "other"


def _policy_input(row: dict[str, Any]) -> dict[str, Any]:
    verification = row.get("verification") or {}
    if not isinstance(verification, dict):
        verification = {}
    return {
        "score": float(verification.get("score", row.get("score", 0.0)) or 0.0),
        "missing_required_elements": list(
            verification.get("missing_required_elements") or []
        ),
        "incorrect_claims": list(verification.get("incorrect_claims") or []),
        "disqualifying_conditions_triggered": list(
            verification.get("disqualifying_conditions_triggered") or []
        ),
    }


def analyze_fail_buckets(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if _row_verdict(row) != "FAIL":
            continue
        domain = _row_domain(row)
        score = _policy_input(row)["score"]
        buckets[domain][_score_bucket(score)] += 1
    return buckets


def compare_modes(
    rows: list[dict[str, Any]], *, fail_only: bool = False
) -> dict[str, Any]:
    flips: list[dict[str, Any]] = []
    strict_pass = 0
    relaxed_pass = 0

    for row in rows:
        stored = _row_verdict(row)
        if fail_only and stored != "FAIL":
            continue
        payload = _policy_input(row)
        strict = apply_verdict_policy(payload, strict=True)
        relaxed = apply_verdict_policy(payload, strict=False)
        if strict["verdict"] == "PASS":
            strict_pass += 1
        if relaxed["verdict"] == "PASS":
            relaxed_pass += 1
        if strict["verdict"] == "FAIL" and relaxed["verdict"] == "PASS":
            flips.append(
                {
                    "id": row.get("id"),
                    "domain": _row_domain(row),
                    "score": payload["score"],
                    "missing_count": len(payload["missing_required_elements"]),
                    "missing": payload["missing_required_elements"],
                }
            )

    total = len(rows) if not fail_only else sum(1 for r in rows if _row_verdict(r) == "FAIL")
    return {
        "rows_scored": total,
        "strict_pass": strict_pass,
        "relaxed_pass": relaxed_pass,
        "flips": flips,
    }


def _print_fail_buckets(buckets: dict[str, Counter[str]]) -> None:
    print("\nFAIL rows by domain and verification.score bucket:")
    print(f"{'domain':<10} {'0.0':>6} {'0.5-0.6':>8} {'0.7+':>6} {'total':>7}")
    print("-" * 42)
    grand = Counter()
    for domain in sorted(buckets):
        counts = buckets[domain]
        total = sum(counts.values())
        grand.update(counts)
        print(
            f"{domain:<10} {counts.get('0.0', 0):>6} "
            f"{counts.get('0.5-0.6', 0):>8} {counts.get('0.7+', 0):>6} {total:>7}"
        )
    print("-" * 42)
    print(
        f"{'ALL':<10} {grand.get('0.0', 0):>6} "
        f"{grand.get('0.5-0.6', 0):>8} {grand.get('0.7+', 0):>6} "
        f"{sum(grand.values()):>7}"
    )


def _print_compare(result: dict[str, Any], *, stored_stats: dict[str, Any]) -> None:
    total = result["rows_scored"]
    strict_pct = 100.0 * result["strict_pass"] / total if total else 0.0
    relaxed_pct = 100.0 * result["relaxed_pass"] / total if total else 0.0
    print("\nJudge mode comparison (re-applied policy on cached verification fields):")
    print(f"  rows scored:   {total}")
    print(f"  strict PASS:   {result['strict_pass']}/{total} ({strict_pct:.1f}%)")
    print(f"  relaxed PASS:  {result['relaxed_pass']}/{total} ({relaxed_pct:.1f}%)")
    print(f"  FAIL→PASS flips (relaxed only): {len(result['flips'])}")
    if stored_stats:
        print(
            f"\nStored experiment verdicts: PASS={stored_stats.get('pass', 0)} "
            f"FAIL={stored_stats.get('fail', 0)} ERROR={stored_stats.get('error', 0)}"
        )
    if result["flips"]:
        print("\nFlipped rows:")
        for flip in result["flips"]:
            missing = flip["missing"] or ["(not stored in artifact)"]
            print(
                f"  - {flip['id']} [{flip['domain']}] score={flip['score']:.2f} "
                f"missing={flip['missing_count']}: {missing[0]}"
            )
    else:
        print(
            "\nNo flips on cached rows. Relaxed mode only helps when score >= 0.75 "
            "with at most one missing element and no incorrect claims."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strict vs relaxed judge policy on a cached experiment."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        type=Path,
        help="Path to experiment dir or experiment.json",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print strict vs relaxed PASS counts and FAIL→PASS flips",
    )
    parser.add_argument(
        "--fail-only",
        action="store_true",
        help="Score only rows whose stored verdict is FAIL",
    )
    args = parser.parse_args()

    if not args.experiment:
        parser.error("experiment path is required")

    path = args.experiment
    if path.is_dir():
        path = path / "experiment.json"
    if not path.is_file():
        print(f"Error: experiment file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = load_experiment(path)
    rows = _rows_from_experiment(data)
    if not rows:
        print("Error: no rows found in experiment artifact.", file=sys.stderr)
        sys.exit(1)

    buckets = analyze_fail_buckets(rows)
    _print_fail_buckets(buckets)

    if args.compare:
        compare = compare_modes(rows, fail_only=args.fail_only)
        _print_compare(compare, stored_stats=pass_stats(data))


if __name__ == "__main__":
    main()
