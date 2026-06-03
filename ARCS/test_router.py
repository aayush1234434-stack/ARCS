"""
Rigorous evaluation for the domain router (DistilBERT classifier).

Usage:
    python test_router.py
    python test_router.py --model-dir ./router-model --output-dir ./eval-results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

DEFAULT_MODEL_DIR = "./router-model"
DEFAULT_TEST_CSV = "router_test.csv"
DEFAULT_TRAIN_CSV = "router_train.csv"
DEFAULT_OUTPUT_DIR = "./eval-results"
MAX_LENGTH = 128
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained domain router.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Held-out test CSV")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, help="Train CSV (overfitting check)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write reports/plots")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-errors", type=int, default=25, help="Misclassified examples to save")
    parser.add_argument("--skip-train", action="store_true", help="Skip train-set evaluation")
    return parser.parse_args()


def load_label_maps(model):
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {k: int(v) for k, v in model.config.label2id.items()}
    class_names = [id2label[i] for i in range(len(id2label))]
    return id2label, label2id, class_names


def load_data(csv_path, label2id):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must have 'text' and 'label' columns")
    unknown = set(df["label"].unique()) - set(label2id.keys())
    if unknown:
        raise ValueError(f"{csv_path} has unknown labels: {unknown}")
    df = df.copy()
    df["label_id"] = df["label"].map(label2id)
    return df


@torch.no_grad()
def predict(model, tokenizer, texts, device, batch_size):
    model.eval()
    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
    }


def confidence_analysis(y_true, y_pred, probs):
    confidences = probs.max(axis=1)
    correct = y_true == y_pred

    bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    bin_labels = ["0.00-0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", "0.95-1.00"]
    bucket_stats = []
    for low, high, label in zip(bins[:-1], bins[1:], bin_labels):
        mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count == 0:
            bucket_stats.append(
                {"bin": label, "count": 0, "accuracy": None, "pct_of_total": 0.0}
            )
            continue
        bucket_stats.append(
            {
                "bin": label,
                "count": count,
                "accuracy": float(correct[mask].mean()),
                "pct_of_total": float(100 * count / len(y_true)),
            }
        )

    return {
        "mean_confidence_all": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(confidences[~correct].mean()) if (~correct).any() else None,
        "high_confidence_errors": int(((~correct) & (confidences >= 0.85)).sum()),
        "low_confidence_correct": int((correct & (confidences < 0.70)).sum()),
        "confidence_bins": bucket_stats,
    }


def collect_errors(df, y_pred, probs, id2label, max_errors):
    errors = df[y_pred != df["label_id"]].copy()
    if errors.empty:
        return []

    errors["predicted"] = y_pred[errors.index]
    errors["predicted_label"] = errors["predicted"].map(id2label)
    errors["confidence"] = probs[errors.index].max(axis=1)
    errors["true_confidence"] = [
        probs[i, int(true_id)] for i, true_id in zip(errors.index, errors["label_id"])
    ]

    errors = errors.sort_values("confidence", ascending=False)
    records = []
    for _, row in errors.head(max_errors).iterrows():
        records.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": round(float(row["confidence"]), 4),
                "true_class_probability": round(float(row["true_confidence"]), 4),
            }
        )
    return records


def plot_confusion_matrix(cm, class_names, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Router confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(name, df, model, tokenizer, device, id2label, class_names, batch_size, max_errors):
    texts = df["text"].tolist()
    y_true = df["label_id"].to_numpy()
    y_pred, probs = predict(model, tokenizer, texts, device, batch_size)

    metrics = compute_metrics(y_true, y_pred, class_names)
    conf = confidence_analysis(y_true, y_pred, probs)
    errors = collect_errors(df.reset_index(drop=True), y_pred, probs, id2label, max_errors)

    return {
        "split": name,
        "samples": len(df),
        "metrics": metrics,
        "confidence": conf,
        "misclassified_examples": errors,
        "num_errors": int((y_pred != y_true).sum()),
    }


def print_report(result, class_names):
    m = result["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  {result['split'].upper()} SET  ({result['samples']} samples)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:           {m['accuracy']:.4f}")
    print(f"  Balanced accuracy:  {m['balanced_accuracy']:.4f}")
    print(f"  Macro F1:           {m['macro_f1']:.4f}")
    print(f"  Weighted F1:        {m['weighted_f1']:.4f}")
    print(f"  Cohen's kappa:      {m['cohen_kappa']:.4f}")
    print(f"  Matthews corr.:     {m['matthews_corrcoef']:.4f}")
    print(f"  Errors:             {result['num_errors']} / {result['samples']}")

    print("\n── Per-class metrics ──")
    for label, stats in m["per_class"].items():
        print(
            f"  {label:8s}  P={stats['precision']:.3f}  "
            f"R={stats['recall']:.3f}  F1={stats['f1']:.3f}  n={stats['support']}"
        )

    print("\n── Classification report ──")
    print(m["classification_report"])

    print("── Confusion matrix (rows=true, cols=pred) ──")
    cm = np.array(m["confusion_matrix"])
    header = "          " + "  ".join(f"{c:>8s}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:8d}" for j in range(len(class_names)))
        print(f"  {row_name:8s}  {row}")

    c = result["confidence"]
    print("\n── Confidence analysis ──")
    print(f"  Mean confidence (all):       {c['mean_confidence_all']:.4f}")
    print(f"  Mean confidence (correct):   {c['mean_confidence_correct']}")
    print(f"  Mean confidence (incorrect): {c['mean_confidence_incorrect']}")
    print(f"  High-confidence errors:      {c['high_confidence_errors']}")
    print(f"  Low-confidence correct:      {c['low_confidence_correct']}")
    print("  Accuracy by confidence bin:")
    for b in c["confidence_bins"]:
        acc = "n/a" if b["accuracy"] is None else f"{b['accuracy']:.3f}"
        print(f"    {b['bin']:12s}  n={b['count']:4d}  acc={acc}  ({b['pct_of_total']:.1f}%)")

    if result["misclassified_examples"]:
        print(f"\n── Top misclassified examples (up to {len(result['misclassified_examples'])}) ──")
        for i, ex in enumerate(result["misclassified_examples"], 1):
            text_preview = ex["text"][:80] + ("..." if len(ex["text"]) > 80 else "")
            print(
                f"  [{i}] true={ex['true_label']}  pred={ex['predicted_label']}  "
                f"conf={ex['confidence']:.3f}  true_prob={ex['true_class_probability']:.3f}"
            )
            print(f"      {text_preview}")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at {model_dir}. Train first: python train_router.py"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    id2label, label2id, class_names = load_label_maps(model)
    print(f"Classes: {class_names}")

    results = []

    test_df = load_data(args.test_csv, label2id)
    test_result = evaluate_split(
        "test", test_df, model, tokenizer, device, id2label, class_names,
        args.batch_size, args.max_errors,
    )
    results.append(test_result)
    print_report(test_result, class_names)

    plot_confusion_matrix(
        np.array(test_result["metrics"]["confusion_matrix"]),
        class_names,
        output_dir / "confusion_matrix_test.png",
    )

    if not args.skip_train:
        train_df = load_data(args.train_csv, label2id)
        train_result = evaluate_split(
            "train", train_df, model, tokenizer, device, id2label, class_names,
            args.batch_size, min(args.max_errors, 10),
        )
        results.append(train_result)
        print_report(train_result, class_names)

        plot_confusion_matrix(
            np.array(train_result["metrics"]["confusion_matrix"]),
            class_names,
            output_dir / "confusion_matrix_train.png",
        )

        gap = train_result["metrics"]["accuracy"] - test_result["metrics"]["accuracy"]
        print(f"\n{'=' * 60}")
        print("  OVERFITTING CHECK (train accuracy − test accuracy)")
        print(f"{'=' * 60}")
        print(f"  Train accuracy: {train_result['metrics']['accuracy']:.4f}")
        print(f"  Test accuracy:  {test_result['metrics']['accuracy']:.4f}")
        print(f"  Gap:            {gap:+.4f}  (large positive ⇒ possible overfitting)")

    # Save machine-readable report
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    errors_path = output_dir / "misclassified_test.json"
    with open(errors_path, "w") as f:
        json.dump(test_result["misclassified_examples"], f, indent=2)

    print(f"\nSaved: {report_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {output_dir / 'confusion_matrix_test.png'}")


if __name__ == "__main__":
    main()
