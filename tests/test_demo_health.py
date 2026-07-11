"""Demo app health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arcs import config
from arcs.demo.app import app
from arcs.router.classifier import ONNX_FILENAME

HAS_ONNX = (config.ROUTER_MODEL_DIR / ONNX_FILENAME).exists()
SKIP_NO_ONNX = pytest.mark.skipif(
    not HAS_ONNX,
    reason="ONNX model not built locally",
)


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "groq_configured": True,
        "nvidia_configured": False,
        "router_backend": "torch",
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


@pytest.mark.integration
@SKIP_NO_ONNX
def test_health_endpoint_reports_onnx_backend(monkeypatch):
    """Health echoes configured backend; ONNX file must exist when backend is onnx."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["router_backend"] == "onnx"
    assert body["groq_configured"] is True
