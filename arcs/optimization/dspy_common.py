"""
Shared DSPy / COPRO helpers for ARCS prompt optimization.

Specialist- and judge-specific modules configure their own LMs where needed
(e.g. NVIDIA for the judge) and call these helpers for Groq setup, COPRO
runs, instruction extraction, and sidecar prompt writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def configure_groq_lm(model: str | None = None) -> Any:
    """Configure DSPy to use Groq via the OpenAI-compatible API."""
    import os

    import dspy
    from dotenv import load_dotenv

    from arcs import config

    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq-backed DSPy optimization")

    lm = dspy.LM(
        model=f"openai/{model or config.resolve_generator_model('MEDICAL')}",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)
    return lm


def run_copro(
    student_module: Any,
    trainset: list[Any],
    metric: Callable[..., Any],
    *,
    breadth: int = 5,
    depth: int = 2,
) -> Any:
    """Compile ``student_module`` with COPRO and return the optimized module."""
    from dspy.teleprompt import COPRO

    optimizer = COPRO(
        metric=metric,
        breadth=breadth,
        depth=depth,
        init_temperature=1.0,
        track_stats=True,
    )
    return optimizer.compile(
        student_module,
        trainset=trainset,
        eval_kwargs={"num_threads": 1},
    )


def extract_instructions(module: Any, fallback_prompt: str) -> str:
    """Pull optimized signature instructions from a DSPy ``Predict`` module."""
    generate = getattr(module, "generate", None)
    if generate is None:
        return fallback_prompt

    signature = getattr(generate, "signature", None)
    if signature is None:
        return fallback_prompt

    instructions = getattr(signature, "instructions", None)
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    return fallback_prompt


def save_sidecar_prompt(
    text: str,
    output_path: Path,
    *,
    component_name: str,
    source_file: str,
) -> Path:
    """Write a review-only sidecar prompt file (does not modify source modules)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Optimized {component_name} system prompt (DSPy COPRO)\n"
        f"# Review this file, then manually replace SYSTEM_PROMPT in\n"
        f"# {source_file} if it looks better.\n"
        f"# Do not auto-apply without human review.\n"
        f"# " + ("-" * 60) + "\n\n"
    )
    output_path.write_text(header + text.strip() + "\n", encoding="utf-8")
    return output_path
