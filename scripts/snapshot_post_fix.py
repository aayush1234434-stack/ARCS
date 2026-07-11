"""Merge the best per-domain post-fix eval runs into one combined snapshot.

Does NOT call any APIs — it only reads previously-saved experiment artifacts and
stitches them together (reusing the dedupe/merge logic in merge_experiments.py),
then reports per-domain PASS deltas against a baseline pipeline run.

Default inputs:
  - Every saved run matching ``post-fix-*-{legal,coding,medical,general}*`` (oldest→newest)
  - Plus ``post-fix-resume-v1`` (or newest ``*post-fix*resume*``) when present, merged last
    so non-ERROR resume rows replace earlier ERROR rows for the same id.

Override with --experiments.

Writes:
    artifacts/experiments/<ts>_post-fix-v2-merged/experiment.json
    artifacts/experiments/<ts>_post-fix-v2-merged/summary.txt
        (table: domain | baseline PASS | post-fix PASS | delta)

Usage:
    python scripts/snapshot_post_fix.py
    python scripts/snapshot_post_fix.py --experiments DIR1 DIR2 ...
    python scripts/snapshot_post_fix.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
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

# Reuse the merge logic from the sibling script (scripts/ is not a package).
_MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_experiments", _ROOT / "scripts" / "merge_experiments.py"
)
merge_experiments = importlib.util.module_from_spec(_MERGE_SPEC)
_MERGE_SPEC.loader.exec_module(merge_experiments)

DEFAULT_BASELINE = (
    config.EXPERIMENTS_DIR / "2026-07-10T07-24-20_baseline-v1-full-pipeline"
)
MERGED_NAME = "post-fix-v2-merged"

# Per-domain discovery: latest dir whose name contains post-fix + domain tag.
DOMAIN_TAGS = ("legal", "coding", "medical", "general")
# Skip combined / non-domain runs when picking per-domain inputs.
_EXCLUDE_DIR_MARKERS = (
    "merged",
    "resume",
    "baseline",
    "post-fix-full",
    "post-fix-clean",
)


def _experiment_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "experiment.json").exists()
    ]


def _matches_domain_dir(name: str, domain_tag: str) -> bool:
    lower = name.lower()
    if "post-fix" not in lower:
        return False
    if domain_tag not in lower:
        return False
    return not any(marker in lower for marker in _EXCLUDE_DIR_MARKERS)


def _all_domain_experiments(domain_tag: str) -> list[Path]:
    """All experiment dirs matching ``post-fix-*-{domain}*``, oldest first."""
    candidates = [
        p for p in _experiment_dirs(config.EXPERIMENTS_DIR)
        if _matches_domain_dir(p.name, domain_tag)
    ]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _latest_resume_experiment() -> Path | None:
    """Latest ``*post-fix*resume*`` experiment (e.g. post-fix-resume-v1)."""
    candidates = [
        p
        for p in _experiment_dirs(config.EXPERIMENTS_DIR)
        if "post-fix" in p.name.lower() and "resume" in p.name.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_default_inputs() -> tuple[list[Path], list[str], Path | None]:
    """Return (per-domain dirs oldest→newest, missing patterns, optional resume dir).

    Includes every ``post-fix-*-{domain}*`` run (not only the newest) so partial
    v2 reruns layer on top of full v1 domain passes. Resume is appended last.
    """
    found: list[Path] = []
    missing: list[str] = []
    for tag in DOMAIN_TAGS:
        matches = _all_domain_experiments(tag)
        pattern = f"post-fix-*-{tag}*"
        if not matches:
            missing.append(pattern)
        else:
            found.extend(matches)
    resume = _latest_resume_experiment()
    if resume is not None:
        found.append(resume)
    return found, missing, resume


def _per_domain_pass(pipeline: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return merge_experiments._per_domain_pass(pipeline)


def _fmt_pass(passed: int, n: int) -> str:
    if n == 0:
        return "-"
    return f"{passed}/{n} ({passed / n:.0%})"


def _build_summary_table(
    merged_pipeline: dict[str, Any],
    baseline_pipeline: dict[str, Any] | None,
) -> str:
    merged = _per_domain_pass(merged_pipeline)
    base = _per_domain_pass(baseline_pipeline) if baseline_pipeline else None

    lines: list[str] = []
    lines.append("Post-fix merged snapshot")
    lines.append("")
    header = f"{'domain':8s} | {'baseline PASS':>14s} | {'post-fix PASS':>14s} | {'delta':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for domain in VALID_DOMAINS:
        mp, mn = merged[domain]
        if base is not None:
            bp, bn = base[domain]
            base_str = _fmt_pass(bp, bn)
            if bn and mn:
                delta = (mp / mn) - (bp / bn)
                delta_str = f"{delta:+.0%}"
            else:
                delta_str = "-"
        else:
            base_str = "-"
            delta_str = "-"
        lines.append(
            f"{domain:8s} | {base_str:>14s} | {_fmt_pass(mp, mn):>14s} | {delta_str:>8s}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge best per-domain post-fix eval runs into one snapshot and "
            "report per-domain PASS deltas vs a baseline. No API calls."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit experiment dirs/JSON to merge (overrides defaults).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline pipeline experiment (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--name",
        default=MERGED_NAME,
        help=f"Merged experiment name (default: {MERGED_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan; no writes.")
    parser.add_argument("--json", action="store_true", help="Print merged dict to stdout.")
    args = parser.parse_args()

    # ── Resolve inputs ──
    resume_path: Path | None = None
    if args.experiments is not None:
        input_paths = list(args.experiments)
        missing_names: list[str] = []
    else:
        domain_paths, missing_names, resume_path = _resolve_default_inputs()
        input_paths = list(domain_paths)
        for pattern in missing_names:
            print(f"Warning: no experiment found for {pattern!r}", file=sys.stderr)
        if resume_path is not None:
            print(f"Including resume experiment: {resume_path}", file=sys.stderr)

    if len(input_paths) < 1:
        print("Error: no input experiments resolved.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(input_paths)} experiment(s):", file=sys.stderr)
    for path in input_paths:
        print(f"  - {path}", file=sys.stderr)

    baseline_exists = args.baseline is not None and Path(
        args.baseline if args.baseline.is_file() else args.baseline / "experiment.json"
    ).exists()
    print(
        f"baseline: {args.baseline}" + ("" if baseline_exists else "  [not found]"),
        file=sys.stderr,
    )

    if args.dry_run:
        print("(dry-run: no merge computed, no artifacts written)", file=sys.stderr)
        return

    # ── Load + merge ──
    loaded: list[tuple[dict[str, Any], Path]] = []
    for path in input_paths:
        try:
            loaded.append((load_experiment(path), path))
        except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
            print(f"Error: could not load {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    loaded.sort(key=lambda pair: merge_experiments._created_key(pair[0], pair[1]))
    experiments = [exp for exp, _ in loaded]
    merged_rows = merge_experiments.merge_experiment_rows(experiments)
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
            "resume_from": str(resume_path) if resume_path is not None else None,
            "baseline": str(args.baseline),
            "n_rows": len(merged_rows),
            "rows": merged_rows,
        },
    )

    # ── Baseline for delta table ──
    baseline_pipeline = None
    if baseline_exists:
        try:
            baseline_pipeline = (load_experiment(args.baseline) or {}).get("pipeline")
        except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load --baseline: {exc}", file=sys.stderr)

    saved_to = save_experiment(experiment, name=args.name, output_dir=args.output_dir)
    experiment = load_experiment(saved_to)

    # Overwrite summary.txt with the requested delta table.
    table = _build_summary_table(experiment.get("pipeline") or pipeline, baseline_pipeline)
    (saved_to / "summary.txt").write_text(table, encoding="utf-8")

    print(f"\nMerged {len(experiments)} run(s) -> {len(merged_rows)} unique row(s)", file=sys.stderr)
    print(table, file=sys.stderr)
    print(f"Saved: {saved_to}", file=sys.stderr)

    if args.json:
        print(json.dumps(experiment, indent=2, default=str))


if __name__ == "__main__":
    main()
