"""CLI: capture a pre-repair baseline (router + pipeline eval) as one manifest.

Runs ``scripts/eval_router.py`` and ``scripts/eval_pipeline.py``, then writes
``artifacts/experiments/<run_id>_baseline/manifest.json`` with pointers and
top-line metrics for later ``compare_experiments.py`` diffs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.eval.experiments import load_experiment, make_run_id

EVAL_ROUTER = _ROOT / "scripts" / "eval_router.py"
EVAL_PIPELINE = _ROOT / "scripts" / "eval_pipeline.py"
DEFAULT_EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"


def _existing_run_dirs(experiments_dir: Path) -> set[Path]:
    if not experiments_dir.exists():
        return set()
    return {p for p in experiments_dir.iterdir() if p.is_dir()}


def _new_run_dir(before: set[Path], after: set[Path], *, name_slug: str) -> Path | None:
    """Pick the new experiment dir that matches the child name slug."""
    created = [p for p in (after - before) if (p / "experiment.json").exists()]
    if not created:
        return None
    # Prefer dirs whose name ends with the slug from make_run_id(name).
    slug = name_slug.strip().lower().replace(" ", "-")
    matches = [p for p in created if p.name.endswith(f"_{slug}") or slug in p.name]
    candidates = matches or created
    return max(candidates, key=lambda p: p.name)


def _run_script(script: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script), *args]
    print(f"\n>>> {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(_ROOT), check=False)
    return result.returncode


def _extract_topline(router_exp: dict[str, Any] | None, pipeline_exp: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "router_test_accuracy": None,
        "eval_queries_router_accuracy": None,
        "pipeline_pass_rate": None,
        "pipeline_router_accuracy": None,
    }

    if isinstance(router_exp, dict):
        router_metrics = router_exp.get("metrics") or {}
        if isinstance(router_metrics, dict):
            metrics["router_test_accuracy"] = router_metrics.get("accuracy")
        eq = router_exp.get("eval_queries_router_accuracy")
        if isinstance(eq, dict):
            eq_metrics = eq.get("metrics") or {}
            if isinstance(eq_metrics, dict):
                metrics["eval_queries_router_accuracy"] = eq_metrics.get("accuracy")

    if isinstance(pipeline_exp, dict):
        pipeline = pipeline_exp.get("pipeline") or {}
        if isinstance(pipeline, dict):
            rates = pipeline.get("status_rates") or {}
            if isinstance(rates, dict):
                metrics["pipeline_pass_rate"] = rates.get("PASS")
        router = pipeline_exp.get("router") or {}
        if isinstance(router, dict):
            metrics["pipeline_router_accuracy"] = router.get("accuracy")

    return metrics


def _write_manifest(
    *,
    name: str,
    manifest_dir: Path,
    router_path: Path | None,
    pipeline_path: Path | None,
    router_exp: dict[str, Any] | None,
    pipeline_exp: dict[str, Any] | None,
    meta: dict[str, Any],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    topline = _extract_topline(router_exp, pipeline_exp)
    payload = {
        "name": name,
        "kind": "baseline_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_dir": str(manifest_dir),
        "children": {
            "router": {
                "path": str(router_path) if router_path else None,
                "name": (router_exp or {}).get("name"),
                "run_id": ((router_exp or {}).get("meta") or {}).get("run_id"),
            },
            "pipeline": {
                "path": str(pipeline_path) if pipeline_path else None,
                "name": (pipeline_exp or {}).get("name"),
                "run_id": ((pipeline_exp or {}).get("meta") or {}).get("run_id"),
            },
        },
        "metrics": topline,
        "meta": meta,
    }
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    # Short human summary beside the JSON.
    lines = [
        f"Baseline snapshot: {name}",
        f"manifest: {path}",
        "",
        "Children:",
        f"  router:   {router_path or '(skipped)'}",
        f"  pipeline: {pipeline_path or '(skipped)'}",
        "",
        "Top-line metrics:",
        f"  router_test_accuracy:          {topline['router_test_accuracy']}",
        f"  eval_queries_router_accuracy:  {topline['eval_queries_router_accuracy']}",
        f"  pipeline_pass_rate:            {topline['pipeline_pass_rate']}",
        f"  pipeline_router_accuracy:      {topline['pipeline_router_accuracy']}",
        "",
    ]
    (manifest_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_next_steps(manifest_path: Path, router_path: Path | None, pipeline_path: Path | None) -> None:
    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("Baseline snapshot complete.", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Use this manifest before any repair; compare with "
        "compare_experiments.py after repair.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    if router_path is not None:
        print(
            f"  python scripts/compare_experiments.py {router_path} "
            "<post-repair-router-run>",
            file=sys.stderr,
        )
    if pipeline_path is not None:
        print(
            f"  python scripts/compare_experiments.py {pipeline_path} "
            "<post-repair-pipeline-run>",
            file=sys.stderr,
        )
    print("  python scripts/compare_experiments.py --list", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a pre-repair baseline by running router + pipeline eval, "
            "then writing a combined manifest under artifacts/experiments/."
        ),
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Baseline name (e.g. baseline-v1). Child runs use <name>-router / <name>-pipeline.",
    )
    parser.add_argument(
        "--pipeline-limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit pipeline eval rows (default: all eval_queries.jsonl rows)",
    )
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="Only run eval_router.py",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Only run eval_pipeline.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands; do not run evals or write a manifest",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--eval-queries",
        type=Path,
        default=DEFAULT_EVAL_QUERIES,
        help=f"Held-out eval queries for router scoring (default: {DEFAULT_EVAL_QUERIES})",
    )
    parser.add_argument(
        "--with-train-eval",
        action="store_true",
        help="Also evaluate the router train split (default: skip train overfit check)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Pass --quiet to eval_pipeline",
    )
    args = parser.parse_args()

    if args.router_only and args.pipeline_only:
        print("Error: choose at most one of --router-only / --pipeline-only", file=sys.stderr)
        sys.exit(1)

    if args.pipeline_limit is not None and args.pipeline_limit < 0:
        print("Error: --pipeline-limit must be >= 0", file=sys.stderr)
        sys.exit(1)

    if not EVAL_ROUTER.exists() or not EVAL_PIPELINE.exists():
        print("Error: eval scripts missing under scripts/", file=sys.stderr)
        sys.exit(1)

    experiments_dir = args.experiments_dir or config.EXPERIMENTS_DIR
    run_router = not args.pipeline_only
    run_pipeline = not args.router_only
    skip_train = not args.with_train_eval  # default: skip train-set overfit check

    router_name = f"{args.name}-router"
    pipeline_name = f"{args.name}-pipeline"
    run_id = make_run_id(args.name)
    manifest_dir = experiments_dir / f"{run_id}_baseline"

    print("Baseline snapshot plan", file=sys.stderr)
    print(f"  name:            {args.name}", file=sys.stderr)
    print(f"  experiments_dir: {experiments_dir}", file=sys.stderr)
    print(f"  manifest_dir:    {manifest_dir}", file=sys.stderr)
    if run_router:
        print(
            f"  [1] eval_router.py --name {router_name} --eval-queries "
            f"{args.eval_queries}"
            + (" --skip-train" if skip_train else ""),
            file=sys.stderr,
        )
    else:
        print("  [1] eval_router.py  (skipped)", file=sys.stderr)
    if run_pipeline:
        limit_note = (
            f" --limit {args.pipeline_limit}" if args.pipeline_limit is not None else ""
        )
        print(
            f"  [2] eval_pipeline.py --name {pipeline_name}{limit_note}"
            + (" --quiet" if args.quiet else ""),
            file=sys.stderr,
        )
    else:
        print("  [2] eval_pipeline.py  (skipped)", file=sys.stderr)

    if args.dry_run:
        print("(dry-run: no evals run, no manifest written)", file=sys.stderr)
        return

    experiments_dir.mkdir(parents=True, exist_ok=True)

    router_path: Path | None = None
    pipeline_path: Path | None = None
    router_exp: dict[str, Any] | None = None
    pipeline_exp: dict[str, Any] | None = None
    errors: list[str] = []

    if run_router:
        before = _existing_run_dirs(experiments_dir)
        router_args = [
            "--name",
            router_name,
            "--eval-queries",
            str(args.eval_queries),
        ]
        if skip_train:
            router_args.append("--skip-train")
        if args.experiments_dir is not None:
            router_args.extend(["--experiments-dir", str(args.experiments_dir)])
        code = _run_script(EVAL_ROUTER, router_args)
        after = _existing_run_dirs(experiments_dir)
        router_path = _new_run_dir(before, after, name_slug=router_name)
        if code != 0:
            errors.append(f"eval_router.py exited {code}")
        if router_path is None:
            errors.append("eval_router.py did not create an experiment directory")
        else:
            try:
                router_exp = load_experiment(router_path)
            except (OSError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"failed to load router experiment: {exc}")

    if run_pipeline:
        before = _existing_run_dirs(experiments_dir)
        pipeline_args = ["--name", pipeline_name]
        if args.pipeline_limit is not None:
            pipeline_args.extend(["--limit", str(args.pipeline_limit)])
        if args.quiet:
            pipeline_args.append("--quiet")
        if args.experiments_dir is not None:
            pipeline_args.extend(["--output-dir", str(args.experiments_dir)])
        code = _run_script(EVAL_PIPELINE, pipeline_args)
        after = _existing_run_dirs(experiments_dir)
        pipeline_path = _new_run_dir(before, after, name_slug=pipeline_name)
        if code != 0:
            errors.append(f"eval_pipeline.py exited {code}")
        if pipeline_path is None:
            errors.append("eval_pipeline.py did not create an experiment directory")
        else:
            try:
                pipeline_exp = load_experiment(pipeline_path)
            except (OSError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"failed to load pipeline experiment: {exc}")

    manifest_path = _write_manifest(
        name=args.name,
        manifest_dir=manifest_dir,
        router_path=router_path,
        pipeline_path=pipeline_path,
        router_exp=router_exp,
        pipeline_exp=pipeline_exp,
        meta={
            "run_id": run_id,
            "pipeline_limit": args.pipeline_limit,
            "router_only": args.router_only,
            "pipeline_only": args.pipeline_only,
            "skip_train": skip_train,
            "eval_queries": str(args.eval_queries),
            "errors": errors,
        },
    )

    topline = _extract_topline(router_exp, pipeline_exp)
    print("", file=sys.stderr)
    print("=== Baseline topline ===", file=sys.stderr)
    for key, value in topline.items():
        print(f"  {key}: {value}", file=sys.stderr)
    print(f"  manifest: {manifest_path}", file=sys.stderr)

    _print_next_steps(manifest_path, router_path, pipeline_path)

    if errors:
        print("\nWarnings/errors during snapshot:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        # Still exit 0 if we wrote a manifest with at least one child — comparison
        # tooling can proceed. Exit 1 only when nothing usable was produced.
        if router_path is None and pipeline_path is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
