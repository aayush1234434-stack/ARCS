"""Groq-safe DSPy LM helpers."""

from __future__ import annotations

import pytest

from arcs.optimization.dspy_common import (
    GROQ_COPRO_DEFAULT_BREADTH,
    clamp_groq_lm_kwargs,
    configure_groq_lm,
    validate_copro_breadth,
)


def test_clamp_groq_lm_kwargs_forces_n_one():
    requested, kwargs = clamp_groq_lm_kwargs({"n": 5, "temperature": 0.7})
    assert requested == 5
    assert kwargs["n"] == 1
    assert "num_generations" not in kwargs


def test_validate_copro_breadth_rejects_one():
    with pytest.raises(ValueError, match="breadth must be > 1"):
        validate_copro_breadth(1)


def test_groq_copro_default_breadth_is_minimum():
    assert GROQ_COPRO_DEFAULT_BREADTH == 2


def test_configure_groq_lm_returns_dspy_lm_subclass(monkeypatch):
    import dspy

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    lm = configure_groq_lm(model="llama-3.3-70b-versatile")
    assert isinstance(lm, dspy.LM)
    assert lm.kwargs.get("n") == 1
