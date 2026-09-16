"""No-verification ablation behavior (no live API calls)."""

from __future__ import annotations

from scripts import eval_no_verification as mod


def test_run_one_routes_and_generates_once_before_posthoc_evaluation(monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(
        mod.router,
        "route",
        lambda query: {
            "domain": "GENERAL",
            "confidence": 0.91,
            "use_fallback": False,
        },
    )

    class Specialist:
        @staticmethod
        def run(query, *, model):
            events.append("generate")
            return {
                "answer": "A test answer.",
                "usage": {"api_calls": 1, "total_tokens": 20},
            }

    class Pipeline:
        pipeline_id = "GENERAL"
        specialist = Specialist()

        @staticmethod
        def resolve_model():
            return "test-model"

    monkeypatch.setattr(mod, "resolve_pipeline", lambda *args, **kwargs: Pipeline())
    monkeypatch.setattr(
        mod.spec_generator,
        "run",
        lambda query: events.append("spec")
        or {
            "intent": "test",
            "required_elements": ["answer"],
            "correctness_criteria": ["correct"],
            "disqualifying_conditions": [],
            "scope": "test",
            "usage": {"api_calls": 1, "total_tokens": 10},
        },
    )
    monkeypatch.setattr(
        mod.judge,
        "run",
        lambda **kwargs: events.append("judge")
        or {
            "verdict": "PASS",
            "score": 0.9,
            "usage": {"api_calls": 1, "total_tokens": 8},
        },
    )

    result = mod._run_one(
        {"id": "sealed-001", "query": "Explain a test.", "expected_domain": "GENERAL"}
    )

    assert events == ["generate", "spec", "judge"]
    assert result["runtime_verification_enabled"] is False
    assert result["status"] == "PASS"
    assert result["usage"]["total"]["api_calls"] == 1
    assert result["usage"]["evaluation"]["total"]["api_calls"] == 2
