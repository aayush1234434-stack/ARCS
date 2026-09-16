"""
Domain router: classify user queries into CODING, MEDICAL, LEGAL, or GENERAL.

Backend is selected by ``ARCS_ROUTER_BACKEND`` (``sklearn`` by default).
``torch`` and ``onnx`` support optional exported DistilBERT checkpoints.
Explicit ``backend=`` on ``route()`` overrides the environment default.

Usage:
    python router.py
    python router.py "What are the symptoms of diabetes?"
    python router.py --backend onnx "..."   # force ONNX
    ARCS_ROUTER_BACKEND=onnx python router.py "..."
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from arcs import config
from arcs import progress

DEFAULT_MODEL_DIR = config.ROUTER_MODEL_DIR
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
_loaded_model_dir: str | None = None

# Artifact-free sklearn backend state. It is fitted deterministically from the
# committed training split on first use and then cached for the process.
_sklearn_pipeline = None
_sklearn_train_fingerprint: str | None = None


def clear_cache() -> None:
    """Drop cached tokenizer/model so the next ``route()`` loads a new checkpoint."""
    global _onnx_session, _tokenizer, _id2label, _model, _device, _loaded_model_dir
    global _sklearn_pipeline, _sklearn_train_fingerprint
    _onnx_session = None
    _tokenizer = None
    _id2label = None
    _model = None
    _device = None
    _loaded_model_dir = None
    _sklearn_pipeline = None
    _sklearn_train_fingerprint = None


def _load_sklearn():
    """Fit the deterministic artifact-free router from the committed train split."""
    global _sklearn_pipeline, _sklearn_train_fingerprint
    if _sklearn_pipeline is not None:
        return _sklearn_pipeline, _sklearn_train_fingerprint

    train_path = config.ROUTER_DATA_DIR / "router_train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"router training split not found: {train_path}")

    texts: list[str] = []
    labels: list[str] = []
    with train_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("text") or "").strip()
            label = str(row.get("label") or "").strip().upper()
            if text and label:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise ValueError(f"router training split is empty: {train_path}")

    progress.log(f"  Fitting sklearn router from {len(texts)} committed examples...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline as SklearnPipeline

    _sklearn_pipeline = SklearnPipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=64.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    _sklearn_pipeline.fit(texts, labels)
    _sklearn_train_fingerprint = hashlib.sha256(train_path.read_bytes()).hexdigest()
    return _sklearn_pipeline, _sklearn_train_fingerprint


def _route_sklearn(query: str) -> dict:
    pipeline, fingerprint = _load_sklearn()
    probabilities = pipeline.predict_proba([query])[0]
    classes = [str(label) for label in pipeline.classes_]
    pred_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    domain = classes[pred_id]
    confidence = float(probabilities[pred_id])
    return {
        "domain": domain,
        "confidence": confidence,
        "all_scores": {
            label: float(probabilities[index]) for index, label in enumerate(classes)
        },
        "use_fallback": confidence < CONFIDENCE_THRESHOLD,
        "model": f"tfidf-logreg-c64 train={fingerprint[:12]}",
        "backend": "sklearn",
    }


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
    global _model, _device, _id2label, _loaded_model_dir

    if _model is not None and _loaded_model_dir == model_dir:
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
    _loaded_model_dir = model_dir

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


def _resolve_backend(backend: str | None, model_dir: str) -> str:
    resolved = (backend or config.ROUTER_BACKEND).strip().lower()
    if resolved not in ("sklearn", "torch", "onnx"):
        raise ValueError(
            "Router backend must be 'sklearn', 'torch', or 'onnx', "
            f"got {resolved!r}"
        )

    if resolved == "onnx" and not (Path(model_dir) / ONNX_FILENAME).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {Path(model_dir) / ONNX_FILENAME}. "
            "Run: python scripts/export_router_onnx.py --model-dir "
            f"{model_dir}"
        )
    return resolved


def route(
    query: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    backend: str | None = None,
) -> dict:
    resolved = _resolve_backend(backend, model_dir)
    if resolved == "sklearn":
        return _route_sklearn(query)
    if resolved == "onnx":
        return _route_onnx(query, model_dir)
    return _route_torch(query, model_dir)


def main():
    parser = argparse.ArgumentParser(description="Route a query to a domain specialist.")
    parser.add_argument("query", nargs="?", help="Query text (prompted if omitted)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model")
    parser.add_argument(
        "--backend",
        choices=("sklearn", "torch", "onnx"),
        default=None,
        help=(
            "Inference backend (default: ARCS_ROUTER_BACKEND env, else torch). "
            "Use onnx in production after export_router_onnx.py."
        ),
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
    labels = {
        "sklearn": "TF-IDF + logistic regression",
        "onnx": "ONNX",
        "torch": "PyTorch + DistilBERT",
    }
    label = labels[backend]

    with progress.step(f"Classify query ({label})"):
        result = route(query, model_dir=args.model_dir, backend=args.backend)

    print(result)


if __name__ == "__main__":
    main()
