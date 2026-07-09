"""
One-time export of the trained router to ONNX for fast inference without PyTorch.

Requires PyTorch (slow on first import). Run once after training, then use
router.py with the ONNX backend for normal pipeline runs.

Usage:
    python export_router_onnx.py
    python export_router_onnx.py --model-dir ./router-model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arcs import config
from arcs import progress

DEFAULT_MODEL_DIR = config.ROUTER_MODEL_DIR
ONNX_FILENAME = "model.onnx"
MAX_LENGTH = 128


def export(model_dir: str) -> Path:
    model_path = Path(model_dir)
    onnx_path = model_path / ONNX_FILENAME

    progress.log("Importing PyTorch for export (one-time; may take a while)...")
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    progress.log(f"Loading model from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    sample = tokenizer(
        "example query for tracing",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    progress.log(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    progress.log(f"Export complete ({size_mb:.1f} MB). Future runs will skip PyTorch.")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export router model to ONNX.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model")
    args = parser.parse_args()

    try:
        with progress.step("Export router to ONNX"):
            export(args.model_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
