"""Train the DistilBERT domain router.

Same DistilBERT setup and metrics as before, now driven by argparse so the
train/test split, output dir, epochs, and seed can be overridden without editing
code. Running with no args reproduces the original behavior exactly.

RQ1 usage (train on a fixed split, write to a run-specific dir; the held-out
test CSV is NEVER modified):

    python -m arcs.router.train \
        --train-csv data/router/router_train.csv \
        --output-dir artifacts/router-model

    # quick experiment with fewer epochs / a different seed
    python -m arcs.router.train --epochs 2 --seed 7 \
        --output-dir artifacts/router-model-seed7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

from arcs import config

# ── Label mapping ──
LABELS = {"CODING": 0, "MEDICAL": 1, "LEGAL": 2, "GENERAL": 3}
ID2LABEL = {v: k for k, v in LABELS.items()}

# Defaults preserve the original hardcoded behavior.
DEFAULT_TRAIN_CSV = config.ROUTER_DATA_DIR / "router_train.csv"
DEFAULT_TEST_CSV = config.ROUTER_DATA_DIR / "router_test.csv"
DEFAULT_OUTPUT_DIR = config.ROUTER_MODEL_DIR
DEFAULT_EPOCHS = 4
DEFAULT_SEED = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


def train(
    *,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> None:
    """Train and save the router. The test CSV is only read, never written."""
    # ── 1. Load data ──
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # ── 2. Label mapping ──
    train_df["label"] = train_df["label"].map(LABELS)
    test_df["label"] = test_df["label"].map(LABELS)

    # ── 3. Tokenizer ──
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    # ── 4. Load model ──
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABELS,
    )

    # ── 5. Training arguments ──
    training_args = TrainingArguments(
        output_dir=str(config.ROUTER_CHECKPOINTS_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        seed=seed,
    )

    # ── 6. Train ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # ── 7. Save model (best checkpoint after load_best_model_at_end) ──
    trainer.save_model(str(output_dir))
    print(f"Model saved to {output_dir}")

    # ── 8. Full evaluation report ──
    print("\n── Final Evaluation ──")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print(
        classification_report(
            labels,
            preds,
            target_names=[ID2LABEL[i] for i in range(len(LABELS))],
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DistilBERT domain router.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"Training CSV (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help=(
            "Held-out test CSV, read-only — NEVER modified by RQ1 "
            f"(default: {DEFAULT_TEST_CSV})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the trained model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
