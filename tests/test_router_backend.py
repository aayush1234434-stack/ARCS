"""Tests for router backend selection."""

from __future__ import annotations

import pytest

from arcs import config
from arcs.router.classifier import _resolve_backend


def test_resolve_backend_defaults_to_config(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)
    assert _resolve_backend(None, str(config.ROUTER_MODEL_DIR)) == "torch"


def test_resolve_backend_env_onnx(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)
    assert _resolve_backend(None, str(config.ROUTER_MODEL_DIR)) == "onnx"


def test_resolve_backend_explicit_onnx():
    assert _resolve_backend("onnx", str(config.ROUTER_MODEL_DIR)) == "onnx"


def test_resolve_backend_onnx_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        _resolve_backend("onnx", str(tmp_path))


def test_resolve_backend_invalid():
    with pytest.raises(ValueError, match="must be 'torch' or 'onnx'"):
        _resolve_backend("auto", str(config.ROUTER_MODEL_DIR))
