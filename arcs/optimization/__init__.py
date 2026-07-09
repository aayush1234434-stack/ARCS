"""
DSPy optimization helpers for ARCS prompt repair.

Supports MEDICAL / CODING / LEGAL / GENERAL specialist, LLM judge (verifier),
and experimental spec-generator prompt sidecars under ``artifacts/prompts/`` —
never overwrites source modules automatically.
"""

from arcs.optimization.metrics import (
    coding_metric,
    judge_metric,
    judge_pass,
    judge_score,
    sandbox_pass,
    spec_metric,
    verifier_false_pass_metric,
)

__all__ = [
    "coding_metric",
    "judge_metric",
    "judge_pass",
    "judge_score",
    "sandbox_pass",
    "spec_metric",
    "verifier_false_pass_metric",
]
