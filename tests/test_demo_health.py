"""Demo app health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from arcs import config
from arcs.demo.app import app


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "groq_configured": True,
        "nvidia_configured": False,
        "router_backend": "onnx",
    }


def test_api_health_alias(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["groq_configured"] is False
    assert body["nvidia_configured"] is True
    assert body["router_backend"] == "torch"
