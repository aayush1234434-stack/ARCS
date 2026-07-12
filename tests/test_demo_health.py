"""Demo app health, rate limits, timeouts, and sanitized errors (no live APIs)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from arcs import config
from arcs.demo import app as demo_app
from arcs.demo.app import InMemoryRateLimiter, app
from arcs.main import PipelineError
from arcs.router.classifier import ONNX_FILENAME

HAS_ONNX = (config.ROUTER_MODEL_DIR / ONNX_FILENAME).exists()
SKIP_NO_ONNX = pytest.mark.skipif(
    not HAS_ONNX,
    reason="ONNX model not built locally",
)


@pytest.fixture(autouse=True)
def _reset_limiters(monkeypatch):
    """Fresh limiters per test so counters do not leak."""
    monkeypatch.setattr(
        demo_app,
        "_query_limiter",
        InMemoryRateLimiter(demo_app.DEMO_QUERY_RATE_LIMIT, demo_app.DEMO_QUERY_RATE_WINDOW),
    )
    monkeypatch.setattr(
        demo_app,
        "_feedback_limiter",
        InMemoryRateLimiter(
            demo_app.DEMO_FEEDBACK_RATE_LIMIT, demo_app.DEMO_FEEDBACK_RATE_WINDOW
        ),
    )


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)
    monkeypatch.setattr(demo_app, "DEMO_PUBLIC", False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["groq_configured"] is True
    assert body["nvidia_configured"] is False
    assert body["router_backend"] == "torch"
    assert body["public_demo"] is False
    assert body["disclaimer"] is None


def test_health_public_demo_disclaimer(monkeypatch):
    monkeypatch.setattr(demo_app, "DEMO_PUBLIC", True)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)

    client = TestClient(app)
    body = client.get("/health").json()
    assert body["public_demo"] is True
    assert body["disclaimer"]
    assert "educational" in body["disclaimer"].lower()


def test_api_health_alias(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)
    monkeypatch.setattr(demo_app, "DEMO_PUBLIC", False)

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["groq_configured"] is False
    assert body["nvidia_configured"] is True
    assert body["router_backend"] == "torch"


def test_query_missing_keys_returns_503_without_key_names_leak(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setattr(
        demo_app, "_query_limiter", InMemoryRateLimiter(100, 60.0)
    )

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "hello"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "API keys" in detail
    assert "GROQ" not in detail  # do not echo which keys are missing


def test_query_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.setattr(demo_app, "_query_limiter", InMemoryRateLimiter(2, 60.0))

    def _fake_pipeline(query: str):
        return {
            "query_id": "q-test",
            "query": query,
            "route": {"domain": "GENERAL", "confidence": 0.9},
            "pipeline": {"pipeline_id": "GENERAL", "verifier": "llm_judge"},
            "specialist": {"answer": "ok"},
            "verification": {"verdict": "PASS", "score": 0.9},
            "timing": {"total_ms": 10},
            "status": "PASS",
        }

    monkeypatch.setattr(demo_app, "run_pipeline", _fake_pipeline)
    monkeypatch.setattr(demo_app, "log_entry", lambda record: {**record, "response": "ok"})

    client = TestClient(app)
    assert client.post("/api/query", json={"query": "one"}).status_code == 200
    assert client.post("/api/query", json={"query": "two"}).status_code == 200
    limited = client.post("/api/query", json={"query": "three"})
    assert limited.status_code == 429
    assert "Rate limit" in limited.json()["detail"]
    assert "Retry-After" in limited.headers


def test_query_timeout_returns_503(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.setattr(demo_app, "_query_limiter", InMemoryRateLimiter(100, 60.0))
    monkeypatch.setattr(demo_app, "DEMO_PIPELINE_TIMEOUT", 0.01)

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(1.0)
        return {}

    monkeypatch.setattr(demo_app, "run_in_threadpool", _hang)

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "slow"})
    assert response.status_code == 503
    assert "timed out" in response.json()["detail"].lower()


def test_pipeline_error_returns_503_without_stack(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.setattr(demo_app, "_query_limiter", InMemoryRateLimiter(100, 60.0))

    def _boom(_query: str):
        raise PipelineError({"error": "secret internals", "error_class": "rate_limit"})

    monkeypatch.setattr(demo_app, "run_pipeline", _boom)

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "boom"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "secret internals" not in detail
    assert "traceback" not in detail.lower()
    assert "unavailable" in detail.lower()


def test_feedback_unknown_query_sanitized(monkeypatch):
    monkeypatch.setattr(demo_app, "_feedback_limiter", InMemoryRateLimiter(100, 60.0))

    def _missing(_qid: str):
        raise LookupError("/abs/path/secrets.jsonl: not found")

    monkeypatch.setattr(demo_app.attribution, "load_record", _missing)

    client = TestClient(app)
    response = client.post(
        "/api/feedback",
        json={"query_id": "missing", "signal": "POSITIVE"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown query_id."
    assert "secrets" not in response.json()["detail"]


def test_rate_limiter_unit():
    limiter = InMemoryRateLimiter(2, 60.0)
    assert limiter.check("ip-a")[0] is True
    assert limiter.check("ip-a")[0] is True
    allowed, retry = limiter.check("ip-a")
    assert allowed is False
    assert retry > 0
    # Other IPs unaffected
    assert limiter.check("ip-b")[0] is True


@pytest.mark.integration
@SKIP_NO_ONNX
def test_health_endpoint_reports_onnx_backend(monkeypatch):
    """Health echoes configured backend; ONNX file must exist when backend is onnx."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)
    monkeypatch.setattr(demo_app, "DEMO_PUBLIC", False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["router_backend"] == "onnx"
    assert body["groq_configured"] is True
