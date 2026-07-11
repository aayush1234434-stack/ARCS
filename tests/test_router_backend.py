"""Tests for router backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from arcs import config
from arcs.router.classifier import ONNX_FILENAME, _resolve_backend


def _touch_onnx(model_dir: Path) -> Path:
    onnx_path = model_dir / ONNX_FILENAME
    onnx_path.write_bytes(b"")
    return onnx_path


def test_resolve_backend_defaults_to_config(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_BACKEND", "torch", raising=False)
    assert _resolve_backend(None, str(config.ROUTER_MODEL_DIR)) == "torch"


def test_resolve_backend_env_onnx(monkeypatch, tmp_path):
    _touch_onnx(tmp_path)
    monkeypatch.setattr(config, "ROUTER_BACKEND", "onnx", raising=False)
    assert _resolve_backend(None, str(tmp_path)) == "onnx"


def test_resolve_backend_explicit_onnx(tmp_path):
    _touch_onnx(tmp_path)
    assert _resolve_backend("onnx", str(tmp_path)) == "onnx"


def test_resolve_backend_onnx_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        _resolve_backend("onnx", str(tmp_path))


def test_resolve_backend_invalid():
    with pytest.raises(ValueError, match="must be 'torch' or 'onnx'"):
        _resolve_backend("auto", str(config.ROUTER_MODEL_DIR))


@pytest.mark.skipif(
    not (config.ROUTER_MODEL_DIR / ONNX_FILENAME).exists(),
    reason="model.onnx not exported under artifacts/router-model/",
)
def test_resolve_backend_real_onnx_export():
    """Optional: validates the checked-in/exported router dir when present locally."""
    assert _resolve_backend("onnx", str(config.ROUTER_MODEL_DIR)) == "onnx"
