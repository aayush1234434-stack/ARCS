"""
Central model, path, and pipeline defaults for ARCS.

A specialist pipeline is defined by prompting strategy, structured output
contract, verification mechanism, and optional toolchain. The underlying
language model is interchangeable via these settings (or environment overrides).
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of the ``arcs`` package).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ROUTER_DATA_DIR = DATA_DIR / "router"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ROUTER_MODEL_DIR = ARTIFACTS_DIR / "router-model"
ROUTER_CHECKPOINTS_DIR = ARTIFACTS_DIR / "router-checkpoints"
EVAL_RESULTS_DIR = ARTIFACTS_DIR / "eval-results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Generator used by domain pipelines when no domain-specific override is set.
DEFAULT_GENERATOR_MODEL = os.getenv(
    "ARCS_GENERATOR_MODEL",
    "llama-3.3-70b-versatile",
)

# Spec generator uses a different family to reduce correlated blind spots.
DEFAULT_SPEC_MODEL = os.getenv(
    "ARCS_SPEC_MODEL",
    "qwen/qwen3-32b",
)

# Independent test-case generator for the coding sandbox (different from generator).
DEFAULT_TEST_GENERATOR_MODEL = os.getenv(
    "ARCS_TEST_GENERATOR_MODEL",
    "qwen/qwen3-32b",
)

# Judge LLM (NVIDIA OpenAI-compatible API).
DEFAULT_JUDGE_MODEL = os.getenv(
    "NVIDIA_JUDGE_MODEL",
    "meta/llama-3.1-8b-instruct",
)

# Optional per-domain generator overrides. Leave unset to use DEFAULT_GENERATOR_MODEL.
DOMAIN_MODEL_OVERRIDES: dict[str, str | None] = {
    "CODING": os.getenv("ARCS_CODING_MODEL") or None,
    "MEDICAL": os.getenv("ARCS_MEDICAL_MODEL") or None,
    "LEGAL": os.getenv("ARCS_LEGAL_MODEL") or None,
    "GENERAL": os.getenv("ARCS_GENERAL_MODEL") or None,
}

CODING_MAX_RETRIES = int(os.getenv("ARCS_CODING_MAX_RETRIES", "3"))
ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("ARCS_ROUTER_CONFIDENCE", "0.75"))


def resolve_generator_model(domain: str) -> str:
    """Return the generator model for a domain, falling back to the default."""
    override = DOMAIN_MODEL_OVERRIDES.get(domain.upper())
    return override or DEFAULT_GENERATOR_MODEL
