"""CLI: evaluate DistilBERT router and save a Phase 2 experiment artifact.

Reuses ``arcs.router.evaluate`` (does not change training). Optionally also
scores ``data/eval_queries.jsonl`` via ``arcs.router.route``.
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
from arcs.eval.metrics import router_accuracy
from arcs.router import evaluate as router_eval


DEFAULT_MODEL_DIR = Path(router_eval.DEFAULT_MODEL_DIR)
DEFAULT_TEST_CSV = Path(router_eval.DEFAULT_TEST_CSV)
DEFAULT_TRAIN_CSV = Path(router_eval.DEFAULT_TRAIN_CSV)
DEFAULT_OUTPUT_DIR = Path(router_eval.DEFAULT_OUTPUT_DIR)
DEFAULT_EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"


def _load_eval_query_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(row, dict):
                continue
            query = row.get("query")
            expected = row.get("expected_domain")
            if not isinstance(query, str) or not query.strip():
                continue
            if not isinstance(expected, str) or not expected.strip():
                continue
            rows.append(row)
    return rows


def _score_eval_queries(
    path: Path,
    *,
    model_dir: str,
) -> dict[str, Any]:
    """Route each held-out eval query and compute router_accuracy."""
    from arcs.router.classifier import route

    rows = _load_eval_query_rows(path)
    scored: list[dict[str, Any]] = []
    for row in rows:
        query = str(row["query"]).strip()
        expected = str(row["expected_domain"]).strip().upper()
        try:
            result = route(query, model_dir=model_dir)
            predicted = str(result.get("domain") or "").strip().upper()
            confidence = result.get("confidence")
        except Exception as exc:  # noqa: BLE001 — keep eval going
            print(
                f"Warning: route failed for {row.get('id')!r}: {exc}",
                file=sys.stderr,
            )
            predicted = None
            confidence = None
        scored.append(
            {
                "id": row.get("id"),
                "query": query,
                "expected_domain": expected,
                "predicted_domain": predicted,
                "router_confidence": confidence,
            }
        )

    metrics = router_accuracy(scored)
    return {
        "source": str(path),
        "n_queries": len(scored),
        "metrics": metrics,
        "rows": scored,
    }


def _normalize_test_metrics(test_result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(test_result.get("metrics") or {})
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "cohen_kappa": metrics.get("cohen_kappa"),
        "matthews_corrcoef": metrics.get("matthews_corrcoef"),
        "per_class": metrics.get("per_class"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "samples": test_result.get("samples"),
        "num_errors": test_result.get("num_errors"),
    }


def _router_summary_block(test_result: dict[str, Any]) -> dict[str, Any]:
    """Shape compatible with ``summary.txt`` router section."""
    metrics = test_result.get("metrics") or {}
    per_class = metrics.get("per_class") or {}
    per_domain_accuracy = {
        label: stats.get("recall")
        for label, stats in per_class.items()
        if isinstance(stats, dict)
    }
    return {
        "n": test_result.get("samples"),
        "correct": (
            None
            if test_result.get("samples") is None or test_result.get("num_errors") is None
            else int(test_result["samples"]) - int(test_result["num_errors"])
        ),
        "accuracy": metrics.get("accuracy"),
        "per_domain_accuracy": per_domain_accuracy,
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
    }


def run_router_evaluation(
    *,
    model_dir: Path,
    test_csv: Path,
    train_csv: Path,
    output_dir: Path,
    skip_train: bool,
    batch_size: int,
    max_errors: int,
) -> dict[str, Any]:
    """Run test (and optional train) evaluation; write eval-results artifacts."""
    import numpy as np
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at {model_dir}. Train first: python -m arcs.router.train"
        )
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not skip_train and not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}", file=sys.stderr)

    tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)

    id2label, label2id, class_names = router_eval.load_label_maps(model)
    print(f"Classes: {class_names}", file=sys.stderr)

    results: list[dict[str, Any]] = []

    test_df = router_eval.load_data(str(test_csv), label2id)
    test_result = router_eval.evaluate_split(
        "test",
        test_df,
        model,
        tokenizer,
        device,
        id2label,
        class_names,
        batch_size,
        max_errors,
    )
    results.append(test_result)
    router_eval.print_report(test_result, class_names)

    cm_test_path = output_dir / "confusion_matrix_test.png"
    router_eval.plot_confusion_matrix(
        np.array(test_result["metrics"]["confusion_matrix"]),
        class_names,
        cm_test_path,
    )

    train_result = None
    overfitting_gap = None
    if not skip_train:
        train_df = router_eval.load_data(str(train_csv), label2id)
        train_result = router_eval.evaluate_split(
            "train",
            train_df,
            model,
            tokenizer,
            device,
            id2label,
            class_names,
            batch_size,
            min(max_errors, 10),
        )
        results.append(train_result)
        router_eval.print_report(train_result, class_names)

        router_eval.plot_confusion_matrix(
            np.array(train_result["metrics"]["confusion_matrix"]),
            class_names,
            output_dir / "confusion_matrix_train.png",
        )

        overfitting_gap = (
            train_result["metrics"]["accuracy"] - test_result["metrics"]["accuracy"]
        )
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("  OVERFITTING CHECK (train accuracy − test accuracy)", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(f"  Train accuracy: {train_result['metrics']['accuracy']:.4f}", file=sys.stderr)
        print(f"  Test accuracy:  {test_result['metrics']['accuracy']:.4f}", file=sys.stderr)
        print(
            f"  Gap:            {overfitting_gap:+.4f}  "
            "(large positive ⇒ possible overfitting)",
            file=sys.stderr,
        )

    report_path = output_dir / "eval_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    errors_path = output_dir / "misclassified_test.json"
    with errors_path.open("w", encoding="utf-8") as fh:
        json.dump(test_result["misclassified_examples"], fh, indent=2)

    print(f"\nSaved: {report_path}", file=sys.stderr)
    print(f"Saved: {errors_path}", file=sys.stderr)
    print(f"Saved: {cm_test_path}", file=sys.stderr)

    return {
        "test_result": test_result,
        "train_result": train_result,
        "overfitting_gap": overfitting_gap,
        "paths": {
            "eval_report": str(report_path),
            "misclassified_test": str(errors_path),
            "confusion_matrix_test": str(cm_test_path),
            "confusion_matrix_train": (
                str(output_dir / "confusion_matrix_train.png") if not skip_train else None
            ),
        },
    }


def build_experiment(
    *,
    name: str,
    model_dir: Path,
    test_csv: Path,
    train_csv: Path,
    eval_output_dir: Path,
    skip_train: bool,
    eval_payload: dict[str, Any],
    eval_queries_result: dict[str, Any] | None,
) -> dict[str, Any]:
    test_result = eval_payload["test_result"]
    metrics = _normalize_test_metrics(test_result)
    experiment: dict[str, Any] = {
        "name": name,
        "kind": "router",
        "metrics": metrics,
        "paths": eval_payload["paths"],
        "router": _router_summary_block(test_result),
        "pipeline": None,
        "meta": {
            "model_dir": str(model_dir),
            "test_csv": str(test_csv),
            "train_csv": str(train_csv),
            "eval_output_dir": str(eval_output_dir),
            "skip_train": skip_train,
            "overfitting_gap": eval_payload.get("overfitting_gap"),
        },
    }
    if eval_queries_result is not None:
        experiment["eval_queries_router_accuracy"] = eval_queries_result
        experiment["meta"]["eval_queries"] = eval_queries_result.get("source")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the DistilBERT domain router and save a normalized "
            "experiment record under artifacts/experiments/."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to saved model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help=f"Held-out test CSV (default: {DEFAULT_TEST_CSV})",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"Train CSV for overfit check (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write eval_report / confusion matrix (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help=f"Experiment artifact root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--name",
        default="router-eval",
        help="Experiment name (default: router-eval)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip train-set evaluation (overfitting check)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=router_eval.BATCH_SIZE,
        help=f"Eval batch size (default: {router_eval.BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=25,
        help="Misclassified examples to save (default: 25)",
    )
    parser.add_argument(
        "--eval-queries",
        type=Path,
        nargs="?",
        const=DEFAULT_EVAL_QUERIES,
        default=None,
        help=(
            "Also score held-out eval queries via arcs.router.route. "
            f"Optional path (default if flag alone: {DEFAULT_EVAL_QUERIES})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print plan; do not load model or write files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print experiment dict to stdout",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write experiment artifacts under artifacts/experiments/",
    )
    args = parser.parse_args()

    missing: list[str] = []
    if not args.model_dir.exists():
        missing.append(f"model-dir: {args.model_dir}")
    if not args.test_csv.exists():
        missing.append(f"test-csv: {args.test_csv}")
    if not args.skip_train and not args.train_csv.exists():
        missing.append(f"train-csv: {args.train_csv}")
    if args.eval_queries is not None and not args.eval_queries.exists():
        missing.append(f"eval-queries: {args.eval_queries}")

    if missing:
        print("Error: missing required path(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)

    print("Router eval plan:", file=sys.stderr)
    print(f"  model_dir:   {args.model_dir}", file=sys.stderr)
    print(f"  test_csv:    {args.test_csv}", file=sys.stderr)
    print(
        f"  train_csv:   {args.train_csv}"
        + (" [skipped]" if args.skip_train else ""),
        file=sys.stderr,
    )
    print(f"  output_dir:  {args.output_dir}", file=sys.stderr)
    print(f"  name:        {args.name}", file=sys.stderr)
    if args.eval_queries is not None:
        print(f"  eval_queries:{args.eval_queries}", file=sys.stderr)

    if args.dry_run:
        print("(dry-run: paths OK — no model load, no artifacts written)", file=sys.stderr)
        plan = {
            "name": args.name,
            "kind": "router",
            "dry_run": True,
            "meta": {
                "model_dir": str(args.model_dir),
                "test_csv": str(args.test_csv),
                "train_csv": str(args.train_csv),
                "eval_output_dir": str(args.output_dir),
                "skip_train": args.skip_train,
                "eval_queries": str(args.eval_queries) if args.eval_queries else None,
            },
        }
        if args.json:
            print(json.dumps(plan, indent=2))
        return

    eval_payload = run_router_evaluation(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        skip_train=args.skip_train,
        batch_size=args.batch_size,
        max_errors=args.max_errors,
    )

    eval_queries_result = None
    if args.eval_queries is not None:
        print(
            f"\nScoring held-out eval queries via arcs.router.route "
            f"({args.eval_queries})...",
            file=sys.stderr,
        )
        eval_queries_result = _score_eval_queries(
            args.eval_queries,
            model_dir=str(args.model_dir),
        )
        acc = (eval_queries_result.get("metrics") or {}).get("accuracy")
        print(
            f"  eval_queries n={eval_queries_result.get('n_queries')}  "
            f"accuracy={acc if acc is None else f'{acc:.3f}'}",
            file=sys.stderr,
        )

    experiment = build_experiment(
        name=args.name,
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        train_csv=args.train_csv,
        eval_output_dir=args.output_dir,
        skip_train=args.skip_train,
        eval_payload=eval_payload,
        eval_queries_result=eval_queries_result,
    )

    saved_to: Path | None = None
    if not args.no_save:
        saved_to = save_experiment(
            experiment,
            name=args.name,
            output_dir=args.experiments_dir,
        )
        experiment = load_experiment(saved_to)
        print(f"\nExperiment saved: {saved_to}", file=sys.stderr)

    metrics = experiment.get("metrics") or {}
    print("", file=sys.stderr)
    print("=== Router experiment summary ===", file=sys.stderr)
    print(f"name:     {experiment.get('name')}", file=sys.stderr)
    print(f"kind:     {experiment.get('kind')}", file=sys.stderr)
    print(
        f"test:     accuracy={metrics.get('accuracy')}  "
        f"macro_f1={metrics.get('macro_f1')}  "
        f"balanced_accuracy={metrics.get('balanced_accuracy')}",
        file=sys.stderr,
    )
    if saved_to is not None:
        print(f"saved:    {saved_to}", file=sys.stderr)

    if args.json:
        print(json.dumps(experiment, indent=2, default=str))


if __name__ == "__main__":
    main()
