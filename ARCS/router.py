import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ── Load once at startup ──
tokenizer = DistilBertTokenizer.from_pretrained("./router-model")
model     = DistilBertForSequenceClassification.from_pretrained("./router-model")
model.eval()

LABEL_MAP = {0: "CODING", 1: "MEDICAL", 2: "LEGAL", 3: "GENERAL"}
CONFIDENCE_THRESHOLD = 0.75

def route(query: str) -> dict:
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = F.softmax(outputs.logits, dim=-1)[0]
    label_idx  = probs.argmax().item()
    confidence = probs[label_idx].item()
    domain     = LABEL_MAP[label_idx]

    # if not confident enough — override to GENERAL
    if confidence < CONFIDENCE_THRESHOLD:
        domain = "GENERAL"

    return {
        "domain":      domain,
        "confidence":  round(confidence, 3),
        "all_scores":  {LABEL_MAP[i]: round(probs[i].item(), 3) for i in range(4)},
        "use_fallback": confidence < CONFIDENCE_THRESHOLD
    }


# ── quick test when you run this file directly ──
if __name__ == "__main__":
    test_queries = [
        "Write a binary search function in Python",
        "What are the side effects of ibuprofen?",
        "Can my landlord evict me without notice in India?",
        "Who invented the telephone?",
        "Explain machine learning",
        "What are the legal implications of a data breach?",
    ]

    for q in test_queries:
        result = route(q)
        fallback = " → FALLBACK" if result["use_fallback"] else ""
        print(f"Q: {q[:55]}")
        print(f"   {result['domain']} ({result['confidence']}){fallback}")
        print()