"""
Domain router: classify user queries into CODING, MEDICAL, LEGAL, or GENERAL.

Usage:
    python router.py
    python router.py "What are the symptoms of diabetes?"
"""

import argparse
import sys

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

DEFAULT_MODEL_DIR = "./router-model"
MAX_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.75

_model = None
_tokenizer = None
_device = None
_id2label = None


def load_model(model_dir=DEFAULT_MODEL_DIR):
    global _model, _tokenizer, _device, _id2label

    if _model is not None:
        return _model, _tokenizer, _device, _id2label

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    _model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}

    return _model, _tokenizer, _device, _id2label


@torch.no_grad()
def route(query, model_dir=DEFAULT_MODEL_DIR):
    model, tokenizer, device, id2label = load_model(model_dir)

    encoded = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred_id = int(torch.argmax(logits, dim=-1).item())
    domain = id2label[pred_id]
    confidence = probs[pred_id]

    all_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "domain": domain,
        "confidence": float(confidence),
        "all_scores": all_scores,
        "use_fallback": confidence < CONFIDENCE_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser(description="Route a query to a domain specialist.")
    parser.add_argument("query", nargs="?", help="Query text (prompted if omitted)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to saved model")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter your query: ").strip()
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            sys.exit(1)

    result = route(query, model_dir=args.model_dir)
    print(result)


if __name__ == "__main__":
    main()
