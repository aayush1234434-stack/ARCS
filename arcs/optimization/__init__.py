"""
DSPy optimization helpers for ARCS prompt repair.

Supports MEDICAL specialist and LLM judge (verifier) prompt sidecars under
``artifacts/prompts/`` — never overwrites source modules automatically.
"""

from arcs.optimization.metrics import (
    judge_metric,
    judge_pass,
    judge_score,
    verifier_false_pass_metric,
)

__all__ = [
    "judge_metric",
    "judge_pass",
    "judge_score",
    "verifier_false_pass_metric",
]
