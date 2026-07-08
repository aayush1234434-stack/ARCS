"""
Domain router: classify user queries into CODING, MEDICAL, LEGAL, or GENERAL.

Uses ONNX Runtime by default when router-model/model.onnx exists (fast startup).
Falls back to PyTorch otherwise.

Usage:
    python router.py
    python router.py "What are the symptoms of diabetes?"
    python router.py --backend torch "..."   # force PyTorch
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import config
import progress

DEFAULT_MODEL_DIR = "./router-model"
ONNX_FILENAME = "model.onnx"
MAX_LENGTH = 128
CONFIDENCE_THRESHOLD = config.ROUTER_CONFIDENCE_THRESHOLD

# ONNX backend state
_onnx_session = None
_tokenizer = None
_id2label = None

# PyTorch backend state
_model = None
_device = None


def _warn_if_disk_full() -> None:
    usage = shutil.disk_usage(".")
    pct = 100 * usage.used / usage.total
    if pct >= 90:
        progress.log(
            f"  WARNING: disk is {pct:.0f}% full — PyTorch imports can take many minutes. "
            "Free space or use ONNX (see export_router_onnx.py)."
        )


def _load_id2label(model_dir: str) -> dict[int, str]:
    config_path = Path(model_dir) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)["id2label"]
    return {int(k): v for k, v in raw.items()}


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _load_tokenizer(model_dir: str):
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    progress.log(f"  Loading tokenizer from {model_dir}...")
    from transformers import DistilBertTokenizer

    _tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    return _tokenizer


def _encode(query: str, model_dir: str) -> dict:
    tokenizer = _load_tokenizer(model_dir)
    return tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def _result_from_logits(
    logits: list[float],
    id2label: dict[int, str],
    *,
    model: str,
    backend: str,
) -> dict:
    probs = _softmax(logits)
    pred_id = max(range(len(probs)), key=probs.__getitem__)
    domain = id2label[pred_id]
    confidence = probs[pred_id]
    all_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "domain": domain,
        "confidence": float(confidence),
        "all_scores": all_scores,
        "use_fallback": confidence < CONFIDENCE_THRESHOLD,
        "model": model,
        "backend": backend,
    }


def _router_model_label(model_dir: str, backend: str) -> str:
    config_path = Path(model_dir) / "config.json"
    base = "distilbert-base-uncased"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base = config.get("_name_or_path") or f"{config.get('model_type', 'distilbert')}-classifier"
        except (OSError, json.JSONDecodeError):
            pass
    if backend == "onnx":
        return f"{base} (onnx)"
    return f"{base} (pytorch)"


def _load_onnx(model_dir: str):
    global _onnx_session, _id2label

    if _onnx_session is not None:
        return _onnx_session, _id2label

    onnx_path = Path(model_dir) / ONNX_FILENAME
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. "
            "Run: python export_router_onnx.py"
        )

    progress.log("  Loading ONNX Runtime...")
    import onnxruntime as ort

    _onnx_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    _id2label = _load_id2label(model_dir)
    return _onnx_session, _id2label


def _route_onnx(query: str, model_dir: str) -> dict:
    import numpy as np

    session, id2label = _load_onnx(model_dir)
    encoded = _encode(query, model_dir)

    logits = session.run(
        None,
        {
            "input_ids": np.array([encoded["input_ids"]], dtype=np.int64),
            "attention_mask": np.array([encoded["attention_mask"]], dtype=np.int64),
        },
    )[0][0]

    return _result_from_logits(
        logits.tolist(),
        id2label,
        model=_router_model_label(model_dir, "onnx"),
        backend="onnx",
    )


def _load_torch(model_dir: str):
    global _model, _device, _id2label

    if _model is not None:
        return _model, _device, _id2label

    _warn_if_disk_full()
    progress.log("  Importing PyTorch (can take 10–30s; if >2 min, free disk space or use ONNX)...")
    import torch
    from transformers import DistilBertForSequenceClassification

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.log(f"  Device: {_device}")

    _load_tokenizer(model_dir)

    progress.log(f"  Loading model weights from {model_dir}...")
    _model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()
    _id2label = _load_id2label(model_dir)

    return _model, _device, _id2label


def _route_torch(query: str, model_dir: str) -> dict:
    import torch

    model, device, id2label = _load_torch(model_dir)
    encoded = _encode(query, model_dir)

    with torch.no_grad():
        input_ids = torch.tensor([encoded["input_ids"]], device=device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_list = logits.squeeze(0).cpu().tolist()

    return _result_from_logits(
        logits_list,
        id2label,
        model=_router_model_label(model_dir, "pytorch"),
        backend="pytorch",
    )


def _resolve_backend(backend: str, model_dir: str) -> str:
    if backend != "auto":
        return backend

    if (Path(model_dir) / ONNX_FILENAME).exists():
        return "onnx"
    return "torch"


def route(query: str, model_dir: str = DEFAULT_MODEL_DIR, backend: str = "auto") -> dict:
    resolved = _resolve_backend(backend, model_dir)
    if resolved == "onnx":
        return _route_onnx(query, model_dir)
    return _route_torch(query, model_dir)


def main():
    parser = argparse.ArgumentParser(description="Route a query to a domain specialist.")
    parser.add_argument("query", nargs="?", help="Query text (prompted if omitted)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model")
    parser.add_argument(
        "--backend",
        choices=("auto", "onnx", "torch"),
        default="auto",
        help="Inference backend (default: onnx if model.onnx exists, else torch)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    args = parser.parse_args()

    progress.set_verbose(not args.quiet)

    query = args.query
    if not query:
        query = input("Enter your query: ").strip()
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            sys.exit(1)

    backend = _resolve_backend(args.backend, args.model_dir)
    label = "ONNX" if backend == "onnx" else "PyTorch + DistilBERT"

    with progress.step(f"Classify query ({label})"):
        result = route(query, model_dir=args.model_dir, backend=args.backend)

    print(result)


if __name__ == "__main__":
    main()
