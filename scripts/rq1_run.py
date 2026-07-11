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
    python scripts/rq1_run.py --execute        # bootstrap corpus (default)
    python scripts/rq1_run.py --execute --corpus real   # RQ1 v2 (real feedback)
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
REAL_CORPUS = config.DATA_DIR / "rq1" / "feedback_corpus_real.jsonl"
EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"

RQ1_V2_MIN_NEGATIVES = 40
RQ1_V2_MIN_ROUTER = 15

LIVE_MODEL_DIR = config.ROUTER_MODEL_DIR
PRE_BACKUP_DIR = config.ARTIFACTS_DIR / "router-model-pre-rq1"
RUN_A_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-a"
RUN_B_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-b"
RUN_A_MODEL_DIR_V2 = config.ARTIFACTS_DIR / "router-model-rq1-v2-run-a"
RUN_B_MODEL_DIR_V2 = config.ARTIFACTS_DIR / "router-model-rq1-v2-run-b"

EVAL_ROUTER = _ROOT / "scripts" / "eval_router.py"


def _resolve_corpus(value: str) -> tuple[Path, str]:
    """Return (corpus path, corpus_kind) where kind is bootstrap|real."""
    if value == "real":
        return REAL_CORPUS, "real"
    if value == "bootstrap":
        return DEFAULT_CORPUS, "bootstrap"
    path = Path(value)
    kind = "real" if "real" in path.name else "bootstrap"
    return path, kind


def _corpus_stats(path: Path) -> tuple[int, int]:
    """Return (total negatives, ROUTER-attributed count) from a corpus JSONL."""
    total = 0
    router = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if (record.get("attribution") or {}).get("component") == "ROUTER":
                router += 1
    return total, router


def _require_real_corpus_ready(corpus_path: Path) -> None:
    total, router = _corpus_stats(corpus_path)
    errors: list[str] = []
    if total < RQ1_V2_MIN_NEGATIVES:
        errors.append(
            f"total negatives {total} < {RQ1_V2_MIN_NEGATIVES} "
            f"(need more live 👎 feedback in logs/requests.jsonl)"
        )
    if router < RQ1_V2_MIN_ROUTER:
        errors.append(
            f"ROUTER-attributed negatives {router} < {RQ1_V2_MIN_ROUTER} "
            f"(need more router-blamed failures with correct_domain labels)"
        )
    if errors:
        print("Error: RQ1 v2 (real feedback) corpus not ready:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            f"\nCorpus: {corpus_path}\n"
            "Build with: python scripts/bootstrap_rq1_corpus.py --real-only\n"
            "Check readiness: python scripts/feedback_stats.py --requests-only",
            file=sys.stderr,
        )
        sys.exit(1)


def _rq1_names(corpus_kind: str) -> dict[str, str]:
    if corpus_kind == "real":
        return {
            "pre": "rq1-v2-pre",
            "run_a": "rq1-v2-run-a",
            "run_b": "rq1-v2-run-b",
            "manifest_slug": "rq1-v2",
        }
    return {
        "pre": "rq1-pre",
        "run_a": "rq1-run-a",
        "run_b": "rq1-run-b",
        "manifest_slug": "rq1",
    }


def _model_dirs(corpus_kind: str) -> tuple[Path, Path]:
    if corpus_kind == "real":
        return RUN_A_MODEL_DIR_V2, RUN_B_MODEL_DIR_V2
    return RUN_A_MODEL_DIR, RUN_B_MODEL_DIR


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


def _eval_queries_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    block = experiment.get("eval_queries_router_accuracy") or {}
    rows = block.get("rows")
    return rows if isinstance(rows, list) else []


def _row_id(row: dict[str, Any]) -> str:
    for key in ("id", "query_id", "query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _is_misroute(row: dict[str, Any]) -> bool:
    expected = str(row.get("expected_domain") or "").strip().upper()
    predicted = row.get("predicted_domain")
    if not expected:
        return False
    if predicted is None:
        return True
    return str(predicted).strip().upper() != expected


def _paired_misroute_comparison(
    pre_exp: dict[str, Any] | None,
    run_a_exp: dict[str, Any],
    run_b_exp: dict[str, Any],
) -> dict[str, Any]:
    """McNemar-style counts on eval-queries misroutes fixed vs pre baseline."""
    pre_rows = {_row_id(r): r for r in _eval_queries_rows(pre_exp or {}) if _row_id(r)}
    a_rows = {_row_id(r): r for r in _eval_queries_rows(run_a_exp) if _row_id(r)}
    b_rows = {_row_id(r): r for r in _eval_queries_rows(run_b_exp) if _row_id(r)}

    ids = sorted(set(pre_rows) | set(a_rows) | set(b_rows))
    pre_misroute_ids = [i for i in ids if pre_rows.get(i) and _is_misroute(pre_rows[i])]

    def _fixed(rows: dict[str, dict[str, Any]], query_id: str) -> bool:
        row = rows.get(query_id)
        return bool(row) and not _is_misroute(row)

    fixed_by_a = sum(1 for i in pre_misroute_ids if _fixed(a_rows, i))
    fixed_by_b = sum(1 for i in pre_misroute_ids if _fixed(b_rows, i))
    b_only = sum(
        1
        for i in pre_misroute_ids
        if _fixed(b_rows, i) and not _fixed(a_rows, i)
    )
    a_only = sum(
        1
        for i in pre_misroute_ids
        if _fixed(a_rows, i) and not _fixed(b_rows, i)
    )

    run_a_misroutes = sum(1 for i in ids if a_rows.get(i) and _is_misroute(a_rows[i]))
    run_b_misroutes = sum(1 for i in ids if b_rows.get(i) and _is_misroute(b_rows[i]))

    note_parts = [
        "McNemar-style paired comparison on eval-queries router misroutes vs pre:",
        f"pre had {len(pre_misroute_ids)} misroute(s);",
        f"Run A fixed {fixed_by_a}, Run B fixed {fixed_by_b};",
        f"discordant pairs (B-only fixes, A-only fixes) = ({b_only}, {a_only}).",
    ]
    if b_only + a_only == 0:
        note_parts.append("Arms agree on every pre misroute (no discordant pairs).")
    elif b_only > a_only:
        note_parts.append("Run B fixed more exclusive pre misroutes than Run A.")
    elif a_only > b_only:
        note_parts.append("Run A fixed more exclusive pre misroutes than Run B.")
    else:
        note_parts.append("Discordant pair counts tied.")

    return {
        "pre_misroute_count": len(pre_misroute_ids),
        "run_a_misroute_count": run_a_misroutes,
        "run_b_misroute_count": run_b_misroutes,
        "misroutes_fixed_by_run_a_vs_pre": fixed_by_a,
        "misroutes_fixed_by_run_b_vs_pre": fixed_by_b,
        "mcnemar_discordant_b_only": b_only,
        "mcnemar_discordant_a_only": a_only,
        "note": " ".join(note_parts),
    }


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


def _print_plan(
    *,
    epochs: int | None,
    skip_pre_eval: bool,
    keep_run_b: bool,
    corpus_path: Path,
    corpus_kind: str,
    names: dict[str, str],
    run_a_model: Path,
    run_b_model: Path,
) -> None:
    label = "RQ1 v2" if corpus_kind == "real" else "RQ1"
    print(f"{label} plan (dry-run — no training). Re-run with --execute to run it.\n")
    print(f"Corpus ({corpus_kind}): {corpus_path}")
    print("\nPrerequisites:")
    for path in (RUN_A_TRAIN, RUN_B_TRAIN):
        mark = "OK" if path.exists() else "MISSING"
        print(f"  [{mark}] {path}")
    backup = "exists" if PRE_BACKUP_DIR.exists() else "will be created from " + str(LIVE_MODEL_DIR)
    print(f"  backup: {PRE_BACKUP_DIR} ({backup})")
    if corpus_kind == "real":
        total, router = _corpus_stats(corpus_path) if corpus_path.exists() else (0, 0)
        ready = total >= RQ1_V2_MIN_NEGATIVES and router >= RQ1_V2_MIN_ROUTER
        print(
            f"  RQ1 v2 thresholds: negatives {total}/{RQ1_V2_MIN_NEGATIVES}, "
            f"ROUTER {router}/{RQ1_V2_MIN_ROUTER} "
            f"({'ready' if ready else 'NOT READY'})"
        )
    print("\nSteps:")
    step = 1
    if not skip_pre_eval:
        print(
            f"  {step}. eval PRE model "
            f"({PRE_BACKUP_DIR if PRE_BACKUP_DIR.exists() else LIVE_MODEL_DIR}) "
            f"-> name {names['pre']}"
        )
        step += 1
    ep = f" --epochs {epochs}" if epochs is not None else ""
    print(
        f"  {step}. train Run A: {RUN_A_TRAIN} -> {run_a_model}{ep}; "
        f"eval -> {names['run_a']}"
    )
    step += 1
    print(
        f"  {step}. train Run B: {RUN_B_TRAIN} -> {run_b_model}{ep}; "
        f"eval -> {names['run_b']}"
    )
    step += 1
    print(
        f"  {step}. write manifest under "
        f"artifacts/experiments/<ts>_{names['manifest_slug']}/manifest.json"
    )
    step += 1
    final = "keep Run B as live model" if keep_run_b else "restore live model from pre-rq1 backup"
    print(f"  {step}. {final}")
    print(
        f"\nAugment rows: run_a={_augment_row_count(RUN_A_AUGMENT)}  "
        f"run_b={_augment_row_count(RUN_B_AUGMENT)}"
    )


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
        default="bootstrap",
        help=(
            'Corpus kind: "bootstrap", "real", or a path to a feedback JSONL '
            f"(default: bootstrap -> {DEFAULT_CORPUS})"
        ),
    )
    parser.add_argument(
        "--keep-run-b",
        action="store_true",
        help="Leave Run B as the live router model instead of restoring the backup.",
    )
    args = parser.parse_args()
    corpus_path, corpus_kind = _resolve_corpus(args.corpus)
    names = _rq1_names(corpus_kind)
    run_a_model_dir, run_b_model_dir = _model_dirs(corpus_kind)

    # --dry-run is explicit and always wins; otherwise dry-run unless --execute.
    if args.dry_run or not args.execute:
        _print_plan(
            epochs=args.epochs,
            skip_pre_eval=args.skip_pre_eval,
            keep_run_b=args.keep_run_b,
            corpus_path=corpus_path,
            corpus_kind=corpus_kind,
            names=names,
            run_a_model=run_a_model_dir,
            run_b_model=run_b_model_dir,
        )
        return

    if corpus_kind == "real":
        if not corpus_path.exists():
            print(f"Error: real corpus not found: {corpus_path}", file=sys.stderr)
            print(
                "Build with: python scripts/bootstrap_rq1_corpus.py --real-only",
                file=sys.stderr,
            )
            sys.exit(1)
        _require_real_corpus_ready(corpus_path)

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
        "corpus_kind": corpus_kind,
        "corpus": str(corpus_path),
        "augment_rows": {
            "run_a": _augment_row_count(RUN_A_AUGMENT),
            "run_b": _augment_row_count(RUN_B_AUGMENT),
        },
        "epochs": args.epochs,
        "experiments": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics: dict[str, dict[str, float | None]] = {}
    pre_exp: dict[str, Any] | None = None

    # ── Step 1: PRE baseline ──
    if not args.skip_pre_eval:
        pre_model = PRE_BACKUP_DIR if PRE_BACKUP_DIR.exists() else LIVE_MODEL_DIR
        pre_dir = _eval_router(pre_model, names["pre"])
        pre_exp = load_experiment(pre_dir)
        metrics["pre"] = {
            "test_acc": _test_acc(pre_exp),
            "eval_queries_acc": _eval_queries_acc(pre_exp),
        }
        manifest["experiments"]["pre"] = str(pre_dir)
    else:
        metrics["pre"] = {"test_acc": None, "eval_queries_acc": None}

    # ── Step 2: Run A ──
    _train(RUN_A_TRAIN, run_a_model_dir, args.epochs)
    run_a_dir = _eval_router(run_a_model_dir, names["run_a"])
    run_a_exp = load_experiment(run_a_dir)
    metrics["run_a"] = {
        "test_acc": _test_acc(run_a_exp),
        "eval_queries_acc": _eval_queries_acc(run_a_exp),
    }
    manifest["experiments"]["run_a"] = str(run_a_dir)

    # ── Step 3: Run B ──
    _train(RUN_B_TRAIN, run_b_model_dir, args.epochs)
    run_b_dir = _eval_router(run_b_model_dir, names["run_b"])
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
    manifest["eval_queries_paired_comparison"] = _paired_misroute_comparison(
        pre_exp, run_a_exp, run_b_exp
    )
    manifest["winner"] = winner
    manifest["winner_criteria"] = "eval_queries accuracy (primary), test accuracy (secondary)"
    manifest["conclusion"] = _conclusion(
        winner, metrics["pre"], metrics["run_a"], metrics["run_b"]
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        + f"_{names['manifest_slug']}"
    )
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
    paired = manifest.get("eval_queries_paired_comparison") or {}
    if paired.get("note"):
        print(f"\nPaired misroutes: {paired['note']}", file=sys.stderr)
    print(f"\nManifest: {manifest_path}", file=sys.stderr)

    # ── Step 5: restore or keep ──
    if args.keep_run_b:
        print(
            f"Installing Run B as live model: {run_b_model_dir} -> {LIVE_MODEL_DIR}",
            file=sys.stderr,
        )
        _copytree(run_b_model_dir, LIVE_MODEL_DIR)
    else:
        print(f"Restoring live model from backup: {PRE_BACKUP_DIR} -> {LIVE_MODEL_DIR}", file=sys.stderr)
        _copytree(PRE_BACKUP_DIR, LIVE_MODEL_DIR)


if __name__ == "__main__":
    main()
