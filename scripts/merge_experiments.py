"""Merge two or more pipeline eval experiments into one combined experiment.

Useful after a run was split across days/quota windows (see eval_pipeline.py
--resume-from): each partial run saved its own experiment, and this stitches
them back into a single authoritative result.

Dedupe rule (per eval id): prefer a non-ERROR row over an ERROR row; between two
rows of the same error-ness, the newer experiment wins. Experiments are ordered
by ``meta.created_at`` (then run_id, then file mtime), oldest to newest.

Metrics (pipeline_summary + router_accuracy) are recomputed from the merged rows
via arcs.eval.metrics, so the combined experiment is internally consistent.

Usage:
    python scripts/merge_experiments.py RUN_DIR_1 RUN_DIR_2 [...] \
        --name post-fix-v2-merged
    python scripts/merge_experiments.py A B --baseline artifacts/experiments/<baseline>
"""

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
from arcs.eval.experiments import load_experiment, save_experiment
from arcs.eval.metrics import (
    VALID_DOMAINS,
    aggregate_experiment,
    pipeline_summary,
    router_accuracy,
)

COMPLETED_STATUSES = frozenset({"PASS", "FAIL"})


def _resolve_json_path(path: Path) -> Path:
    return path / "experiment.json" if path.is_dir() else path


def _created_key(experiment: dict[str, Any], path: Path) -> str:
    meta = experiment.get("meta") or {}
    return str(
        meta.get("created_at")
        or meta.get("run_id")
        or _resolve_json_path(path).stat().st_mtime
    )


def _rows_of(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (experiment.get("meta") or {}).get("rows") or []
    return [r for r in rows if isinstance(r, dict) and r.get("id")]


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "").upper()


def _is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    """Should ``candidate`` (from a newer experiment) replace ``current``?

    Non-ERROR beats ERROR. If both are non-ERROR (or both ERROR), the newer one
    wins — and candidate is always the newer here by iteration order.
    """
    cand_error = _status(candidate) not in COMPLETED_STATUSES
    curr_error = _status(current) not in COMPLETED_STATUSES
    if curr_error and not cand_error:
        return True  # candidate completed, current did not
    if not curr_error and cand_error:
        return False  # current completed, do not regress to an ERROR
    return True  # same error-ness -> newer (candidate) wins


def merge_experiment_rows(
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge rows from experiments ordered oldest->newest, deduped by id.

    Pure function (no I/O) so it is easy to unit test.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for experiment in experiments:
        for row in _rows_of(experiment):
            rid = str(row["id"])
            if rid not in merged:
                merged[rid] = row
                order.append(rid)
            elif _is_better(row, merged[rid]):
                merged[rid] = row
    return [merged[rid] for rid in order]


def _per_domain_pass(pipeline: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """domain -> (pass_count, n) from a pipeline_summary dict."""
    out: dict[str, tuple[int, int]] = {}
    per_domain = pipeline.get("per_domain") or {}
    for domain in VALID_DOMAINS:
        bucket = per_domain.get(domain) or {}
        n = int(bucket.get("n") or 0)
        passed = int((bucket.get("status_counts") or {}).get("PASS", 0))
        out[domain] = (passed, n)
    return out


def _fmt_pass(passed: int, n: int) -> str:
    if n == 0:
        return "   -   "
    return f"{passed}/{n} ({passed / n:.0%})"


def _print_domain_table(
    pipeline: dict[str, Any],
    baseline_pipeline: dict[str, Any] | None,
) -> None:
    merged = _per_domain_pass(pipeline)
    base = _per_domain_pass(baseline_pipeline) if baseline_pipeline else None

    print("\nPer-domain PASS rate", file=sys.stderr)
    if base is None:
        print(f"  {'domain':8s} {'merged':>14s}", file=sys.stderr)
        for domain in VALID_DOMAINS:
            p, n = merged[domain]
            print(f"  {domain:8s} {_fmt_pass(p, n):>14s}", file=sys.stderr)
    else:
        print(
            f"  {'domain':8s} {'baseline':>14s} {'merged':>14s}",
            file=sys.stderr,
        )
        for domain in VALID_DOMAINS:
            bp, bn = base[domain]
            mp, mn = merged[domain]
            print(
                f"  {domain:8s} {_fmt_pass(bp, bn):>14s} {_fmt_pass(mp, mn):>14s}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge 2+ pipeline eval experiments into one combined experiment.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Experiment run dirs or experiment.json files (2 or more).",
    )
    parser.add_argument(
        "--name",
        default="merged-experiment",
        help="Name for the merged experiment (default: merged-experiment)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline experiment to show per-domain PASS deltas against.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the merged experiment; print summary only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the merged experiment dict to stdout.",
    )
    args = parser.parse_args()

    if len(args.paths) < 2:
        print("Error: provide at least 2 experiments to merge.", file=sys.stderr)
        sys.exit(1)

    loaded: list[tuple[dict[str, Any], Path]] = []
    for path in args.paths:
        try:
            loaded.append((load_experiment(path), path))
        except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
            print(f"Error: could not load {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    # Oldest -> newest so later experiments win ties.
    loaded.sort(key=lambda pair: _created_key(pair[0], pair[1]))
    experiments = [exp for exp, _ in loaded]

    merged_rows = merge_experiment_rows(experiments)
    if not merged_rows:
        print("Error: no rows found in the provided experiments.", file=sys.stderr)
        sys.exit(1)

    router = router_accuracy(merged_rows)
    pipeline = pipeline_summary(merged_rows)
    experiment = aggregate_experiment(
        args.name,
        router=router,
        pipeline=pipeline,
        meta={
            "merged_from": [str(p) for _, p in loaded],
            "n_rows": len(merged_rows),
            "rows": merged_rows,
        },
    )

    saved_to: Path | None = None
    if not args.no_save:
        saved_to = save_experiment(experiment, name=args.name, output_dir=args.output_dir)
        experiment = load_experiment(saved_to)

    # ── Summary ──
    print(f"Merged {len(experiments)} experiment(s) -> {len(merged_rows)} unique row(s)", file=sys.stderr)
    status_counts = (experiment.get("pipeline") or {}).get("status_counts") or {}
    print(
        f"status: PASS={status_counts.get('PASS', 0)}  FAIL={status_counts.get('FAIL', 0)}  "
        f"UNKNOWN={status_counts.get('UNKNOWN', 0)}  ERROR={status_counts.get('ERROR', 0)}",
        file=sys.stderr,
    )

    baseline_pipeline = None
    if args.baseline is not None:
        try:
            baseline_pipeline = (load_experiment(args.baseline) or {}).get("pipeline")
        except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load --baseline: {exc}", file=sys.stderr)

    _print_domain_table(experiment.get("pipeline") or pipeline, baseline_pipeline)

    if saved_to is not None:
        print(f"\nSaved: {saved_to}", file=sys.stderr)

    if args.json:
        print(json.dumps(experiment, indent=2, default=str))


if __name__ == "__main__":
    main()
