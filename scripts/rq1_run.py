"""RQ1 orchestrator: does attribution-filtered feedback beat using all feedback?

Compares two router retraining strategies against a frozen pre-RQ1 baseline:

  - Run A: augment the base train set with ALL negative feedback.
  - Run B: augment ONLY with feedback attributed to the ROUTER component.

Hypothesis (RQ1): routing quality improves more when the router is retrained on
feedback that attribution actually blames on the router (Run B) than on every
negative regardless of root cause (Run A) — i.e. targeted repair > blanket repair.

The held-out test set (data/router/router_test.csv) is never modified. Training
and evaluation are delegated to existing tools (``arcs.router.train`` and
``scripts/eval_router.py``) via subprocess so each model is evaluated in a fresh
process (the router caches its model in module globals).

Usage:
    python scripts/rq1_run.py                 # dry-run: prints the plan only
    python scripts/rq1_run.py --execute        # actually train + evaluate
    python scripts/rq1_run.py --execute --epochs 4 --keep-run-b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.eval.experiments import load_experiment

# ── Paths ──
RQ1_DATA_DIR = config.ROUTER_DATA_DIR / "rq1"
RUN_A_TRAIN = RQ1_DATA_DIR / "run_a_train.csv"
RUN_B_TRAIN = RQ1_DATA_DIR / "run_b_train.csv"
RUN_A_AUGMENT = RQ1_DATA_DIR / "run_a_augment.csv"
RUN_B_AUGMENT = RQ1_DATA_DIR / "run_b_augment.csv"
DEFAULT_CORPUS = config.DATA_DIR / "rq1" / "feedback_corpus.jsonl"
EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"

LIVE_MODEL_DIR = config.ROUTER_MODEL_DIR
PRE_BACKUP_DIR = config.ARTIFACTS_DIR / "router-model-pre-rq1"
RUN_A_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-a"
RUN_B_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-b"

EVAL_ROUTER = _ROOT / "scripts" / "eval_router.py"


def _env() -> dict[str, str]:
    """Env for subprocesses; give matplotlib a writable cache dir."""
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
    return env


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(_ROOT), env=_env())
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _augment_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        lines = sum(1 for line in fh if line.strip())
    return max(0, lines - 1)  # minus header


def _snapshot_experiments() -> set[str]:
    root = config.EXPERIMENTS_DIR
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _new_experiment_dir(before: set[str], name_slug: str) -> Path:
    """Find the experiment dir created since ``before`` matching the name slug."""
    root = config.EXPERIMENTS_DIR
    new = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name not in before
        and p.name.endswith(f"_{name_slug}")
        and (p / "experiment.json").exists()
    ]
    if not new:
        raise FileNotFoundError(
            f"no new experiment dir ending in _{name_slug} was created"
        )
    return max(new, key=lambda p: p.name)


def _eval_router(model_dir: Path, name: str) -> Path:
    """Run scripts/eval_router.py for a model; return its saved experiment dir."""
    before = _snapshot_experiments()
    _run(
        [
            sys.executable,
            str(EVAL_ROUTER),
            "--model-dir",
            str(model_dir),
            "--name",
            name,
            "--eval-queries",
            str(EVAL_QUERIES),
            "--skip-train",
        ]
    )
    return _new_experiment_dir(before, name)


def _train(train_csv: Path, output_dir: Path, epochs: int | None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "arcs.router.train",
        "--train-csv",
        str(train_csv),
        "--output-dir",
        str(output_dir),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    _run(cmd)


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _test_acc(experiment: dict[str, Any]) -> float | None:
    metrics = experiment.get("metrics") or {}
    value = metrics.get("accuracy")
    return None if value is None else float(value)


def _eval_queries_acc(experiment: dict[str, Any]) -> float | None:
    block = experiment.get("eval_queries_router_accuracy") or {}
    metrics = block.get("metrics") or {}
    value = metrics.get("accuracy")
    return None if value is None else float(value)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pick_winner(run_a: dict[str, float | None], run_b: dict[str, float | None]) -> str:
    """Winner by eval_queries accuracy (primary), then test accuracy (secondary)."""
    for key in ("eval_queries_acc", "test_acc"):
        a_val, b_val = run_a[key], run_b[key]
        if a_val is None or b_val is None:
            continue
        if a_val > b_val:
            return "run_a"
        if b_val > a_val:
            return "run_b"
    return "tie"


def _conclusion(
    winner: str,
    pre: dict[str, float | None],
    run_a: dict[str, float | None],
    run_b: dict[str, float | None],
) -> str:
    if winner == "run_b":
        lead = (
            "Run B (attribution-filtered feedback) won, supporting the RQ1 "
            "hypothesis that targeted router repair beats blanket repair."
        )
    elif winner == "run_a":
        lead = (
            "Run A (all negative feedback) won, which does NOT support the RQ1 "
            "hypothesis: blanket retraining outperformed attribution-filtered "
            "retraining here."
        )
    else:
        lead = (
            "Run A and Run B tied on both metrics, so this run does not "
            "distinguish targeted from blanket router repair."
        )
    return (
        f"{lead} On the held-out eval queries, router accuracy moved from "
        f"{_fmt(pre['eval_queries_acc'])} (pre) to {_fmt(run_a['eval_queries_acc'])} "
        f"(Run A) and {_fmt(run_b['eval_queries_acc'])} (Run B). On the frozen "
        f"router test set, accuracy moved from {_fmt(pre['test_acc'])} (pre) to "
        f"{_fmt(run_a['test_acc'])} (Run A) and {_fmt(run_b['test_acc'])} (Run B)."
    )


def _print_plan(epochs: int | None, skip_pre_eval: bool, keep_run_b: bool) -> None:
    print("RQ1 plan (dry-run — no training). Re-run with --execute to run it.\n")
    print("Prerequisites:")
    for path in (RUN_A_TRAIN, RUN_B_TRAIN):
        mark = "OK" if path.exists() else "MISSING"
        print(f"  [{mark}] {path}")
    backup = "exists" if PRE_BACKUP_DIR.exists() else "will be created from " + str(LIVE_MODEL_DIR)
    print(f"  backup: {PRE_BACKUP_DIR} ({backup})")
    print("\nSteps:")
    step = 1
    if not skip_pre_eval:
        print(f"  {step}. eval PRE model ({PRE_BACKUP_DIR if PRE_BACKUP_DIR.exists() else LIVE_MODEL_DIR}) -> name rq1-pre")
        step += 1
    ep = f" --epochs {epochs}" if epochs is not None else ""
    print(f"  {step}. train Run A: {RUN_A_TRAIN} -> {RUN_A_MODEL_DIR}{ep}; eval -> rq1-run-a")
    step += 1
    print(f"  {step}. train Run B: {RUN_B_TRAIN} -> {RUN_B_MODEL_DIR}{ep}; eval -> rq1-run-b")
    step += 1
    print(f"  {step}. write manifest under artifacts/experiments/<ts>_rq1/manifest.json")
    step += 1
    final = "keep Run B as live model" if keep_run_b else "restore live model from pre-rq1 backup"
    print(f"  {step}. {final}")
    print(f"\nAugment rows: run_a={_augment_row_count(RUN_A_AUGMENT)}  run_b={_augment_row_count(RUN_B_AUGMENT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 router repair orchestrator.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train and evaluate (default is a dry-run plan only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only (default). Overrides --execute if both are given.",
    )
    parser.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="Skip evaluating the pre-RQ1 baseline model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs passed to arcs.router.train (default: script default).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Feedback corpus path recorded in the manifest (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--keep-run-b",
        action="store_true",
        help="Leave Run B as the live router model instead of restoring the backup.",
    )
    args = parser.parse_args()

    # --dry-run is explicit and always wins; otherwise dry-run unless --execute.
    if args.dry_run or not args.execute:
        _print_plan(args.epochs, args.skip_pre_eval, args.keep_run_b)
        return

    # ── Prerequisites ──
    missing = [str(p) for p in (RUN_A_TRAIN, RUN_B_TRAIN) if not p.exists()]
    if missing:
        print("Error: missing prepared dataset(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("Run scripts/rq1_prepare_datasets.py first.", file=sys.stderr)
        sys.exit(1)

    # Backup the live model so we can restore it afterwards.
    if not PRE_BACKUP_DIR.exists():
        if not LIVE_MODEL_DIR.exists():
            print(
                f"Error: no live model at {LIVE_MODEL_DIR} to back up or evaluate.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Backing up {LIVE_MODEL_DIR} -> {PRE_BACKUP_DIR}", file=sys.stderr)
        _copytree(LIVE_MODEL_DIR, PRE_BACKUP_DIR)

    manifest: dict[str, Any] = {
        "hypothesis": (
            "Router accuracy improves more when retrained on ROUTER-attributed "
            "feedback (Run B) than on all negative feedback (Run A): targeted "
            "repair beats blanket repair."
        ),
        "corpus": str(args.corpus),
        "augment_rows": {
            "run_a": _augment_row_count(RUN_A_AUGMENT),
            "run_b": _augment_row_count(RUN_B_AUGMENT),
        },
        "epochs": args.epochs,
        "experiments": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics: dict[str, dict[str, float | None]] = {}

    # ── Step 1: PRE baseline ──
    if not args.skip_pre_eval:
        pre_model = PRE_BACKUP_DIR if PRE_BACKUP_DIR.exists() else LIVE_MODEL_DIR
        pre_dir = _eval_router(pre_model, "rq1-pre")
        pre_exp = load_experiment(pre_dir)
        metrics["pre"] = {
            "test_acc": _test_acc(pre_exp),
            "eval_queries_acc": _eval_queries_acc(pre_exp),
        }
        manifest["experiments"]["pre"] = str(pre_dir)
    else:
        metrics["pre"] = {"test_acc": None, "eval_queries_acc": None}

    # ── Step 2: Run A ──
    _train(RUN_A_TRAIN, RUN_A_MODEL_DIR, args.epochs)
    run_a_dir = _eval_router(RUN_A_MODEL_DIR, "rq1-run-a")
    run_a_exp = load_experiment(run_a_dir)
    metrics["run_a"] = {
        "test_acc": _test_acc(run_a_exp),
        "eval_queries_acc": _eval_queries_acc(run_a_exp),
    }
    manifest["experiments"]["run_a"] = str(run_a_dir)

    # ── Step 3: Run B ──
    _train(RUN_B_TRAIN, RUN_B_MODEL_DIR, args.epochs)
    run_b_dir = _eval_router(RUN_B_MODEL_DIR, "rq1-run-b")
    run_b_exp = load_experiment(run_b_dir)
    metrics["run_b"] = {
        "test_acc": _test_acc(run_b_exp),
        "eval_queries_acc": _eval_queries_acc(run_b_exp),
    }
    manifest["experiments"]["run_b"] = str(run_b_dir)

    # ── Step 4: manifest ──
    winner = _pick_winner(metrics["run_a"], metrics["run_b"])
    manifest["metrics"] = {
        arm: {
            "router_test_accuracy": vals["test_acc"],
            "eval_queries_router_accuracy": vals["eval_queries_acc"],
        }
        for arm, vals in metrics.items()
    }
    manifest["winner"] = winner
    manifest["winner_criteria"] = "eval_queries accuracy (primary), test accuracy (secondary)"
    manifest["conclusion"] = _conclusion(
        winner, metrics["pre"], metrics["run_a"], metrics["run_b"]
    )

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + "_rq1"
    manifest_dir = config.EXPERIMENTS_DIR / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # ── Metrics table ──
    print("\n=== RQ1 metrics ===", file=sys.stderr)
    header = f"{'arm':6s} {'test_acc':>10s} {'eval_q_acc':>12s}"
    print(header, file=sys.stderr)
    for arm in ("pre", "run_a", "run_b"):
        vals = metrics[arm]
        print(
            f"{arm:6s} {_fmt(vals['test_acc']):>10s} {_fmt(vals['eval_queries_acc']):>12s}",
            file=sys.stderr,
        )
    print(f"winner: {winner}", file=sys.stderr)
    print(f"\nManifest: {manifest_path}", file=sys.stderr)

    # ── Step 5: restore or keep ──
    if args.keep_run_b:
        print(f"Installing Run B as live model: {RUN_B_MODEL_DIR} -> {LIVE_MODEL_DIR}", file=sys.stderr)
        _copytree(RUN_B_MODEL_DIR, LIVE_MODEL_DIR)
    else:
        print(f"Restoring live model from backup: {PRE_BACKUP_DIR} -> {LIVE_MODEL_DIR}", file=sys.stderr)
        _copytree(PRE_BACKUP_DIR, LIVE_MODEL_DIR)


if __name__ == "__main__":
    main()
