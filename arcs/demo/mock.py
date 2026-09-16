"""Deterministic offline pipeline used to tour the demo without API keys."""

from __future__ import annotations

import re
from uuid import uuid4

DOMAIN_TERMS = {
    "CODING": ("python", "code", "function", "api", "sql", "javascript", "debug"),
    "MEDICAL": ("symptom", "dose", "doctor", "medical", "pain", "medicine", "health"),
    "LEGAL": ("legal", "law", "landlord", "contract", "court", "rights", "attorney"),
}

SAMPLE_ANSWERS = {
    "CODING": (
        "```python\ndef reverse_string(value: str) -> str:\n"
        "    return value[::-1]\n```\n\n"
        "This uses Python slicing, runs in O(n) time, and handles the empty string."
    ),
    "MEDICAL": (
        "This offline example cannot assess an individual condition. Warning signs such as "
        "severe or worsening pain, breathing difficulty, confusion, fainting, or dehydration "
        "need prompt medical assessment. Contact a qualified clinician for advice based on age, "
        "history, medications, and symptoms; use emergency services for severe symptoms."
    ),
    "LEGAL": (
        "This is general legal information, not legal advice. The applicable rule depends on the "
        "jurisdiction, contract wording, notice requirements, and available exceptions. Preserve "
        "the relevant documents and dates, check the official local rule, and consult a licensed "
        "lawyer before acting on a deadline or material right."
    ),
    "GENERAL": (
        "This is a deterministic offline demonstration of ARCS. In live mode, the query is routed "
        "to a domain pipeline, answered by a configured model, checked against an independently "
        "generated specification, and logged with verification and feedback metadata."
    ),
}


def _route(query: str) -> tuple[str, float]:
    text = query.lower()
    scores = {
        domain: sum(bool(re.search(rf"\b{re.escape(term)}\b", text)) for term in terms)
        for domain, terms in DOMAIN_TERMS.items()
    }
    domain, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "GENERAL", 0.62
    return domain, min(0.98, 0.72 + 0.08 * score)


def run_pipeline(query: str) -> dict:
    """Return a realistic, explicitly marked pipeline record with no network calls."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    domain, confidence = _route(query)
    verifier = "sandbox" if domain == "CODING" else "llm_judge"
    answer = SAMPLE_ANSWERS[domain]
    return {
        "query_id": str(uuid4()),
        "query": query,
        "route": {
            "domain": domain,
            "confidence": confidence,
            "use_fallback": domain == "GENERAL",
            "backend": "offline-rules",
        },
        "pipeline": {
            "pipeline_id": domain,
            "verifier": verifier,
            "generator_model": "offline/deterministic-demo",
            "tools": ["sandbox"] if domain == "CODING" else [],
            "max_retries": 1,
        },
        "specification": {
            "intent": "Demonstrate the ARCS execution trace without external services.",
            "required_elements": ["A direct answer", "Scope or safety caveats"],
            "correctness_criteria": ["The response is clearly marked as offline demo output"],
            "disqualifying_conditions": ["Claims that a live model or sandbox ran"],
            "scope": "Product-tour output only; not an evaluation result.",
            "model": "offline/deterministic-demo",
        },
        "specialist": {
            "answer": answer,
            "domain": domain,
            "specialist": "offline/deterministic-demo",
            "generator_model": "offline/deterministic-demo",
            "pipeline_id": domain,
        },
        "verification": {
            "verification_type": "OFFLINE_DEMO",
            "verdict": "PASS",
            "score": 1.0,
            "explanation": "Deterministic demo fixture validated; no live model was called.",
            "model": "offline/deterministic-demo",
        },
        "tooling": {
            "offline_demo": True,
            "delivery_warning": "Offline demo mode: no live model or code sandbox ran.",
            "rounds_used": 1,
        },
        "timing": {
            "route_ms": 1,
            "specification_ms": 1,
            "specialist_ms": 2,
            "verification_ms": 1,
            "total_ms": 5,
        },
        "usage": {
            "total": {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "components": {},
            "cost_usd": 0.0,
            "cost_note": "Offline deterministic demo; no provider calls.",
        },
        "demo_mode": "offline",
    }
