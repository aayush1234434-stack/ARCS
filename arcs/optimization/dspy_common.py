"""
Shared DSPy / COPRO helpers for ARCS prompt optimization.

Specialist- and judge-specific modules configure their own LMs where needed
(e.g. NVIDIA for the judge) and call these helpers for Groq setup, COPRO
runs, instruction extraction, and sidecar prompt writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

# Groq chat completions reject n>1. DSPy COPRO 3.2 requires breadth>1, so we
# default to the minimum and emulate multi-completion requests sequentially.
GROQ_COPRO_DEFAULT_BREADTH = 2
GROQ_COPRO_DEFAULT_DEPTH = 2


def validate_copro_breadth(breadth: int) -> None:
    """COPRO raises if breadth <= 1; GroqSafeLM handles n=1 at the API."""
    if breadth <= 1:
        raise ValueError(
            "COPRO breadth must be > 1 (DSPy 3.2). Use --breadth 2 or higher. "
            "Groq accepts n=1 per request only; GroqSafeLM emulates n>1 "
            "with sequential calls when COPRO requests multiple completions."
        )


def clamp_groq_lm_kwargs(kwargs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Return (requested_n, kwargs_with_n_forced_to_1) for Groq API calls."""
    merged = dict(kwargs)
    requested = merged.pop("n", None) or merged.pop("num_generations", None) or 1
    try:
        requested_n = max(1, int(requested))
    except (TypeError, ValueError):
        requested_n = 1
    merged["n"] = 1
    merged.pop("num_generations", None)
    return requested_n, merged


def _response_choices(response: Any) -> list[Any]:
    if response is None:
        return []
    choices = getattr(response, "choices", None)
    if choices is not None:
        return list(choices)
    if isinstance(response, dict):
        return list(response.get("choices") or [])
    return []


def _merge_response_choices(base: Any, choices: list[Any]) -> Any:
    if not choices:
        return base
    if hasattr(base, "choices"):
        base.choices = choices
        return base
    if isinstance(base, dict):
        merged = dict(base)
        merged["choices"] = choices
        return merged
    return base


def configure_groq_lm(model: str | None = None) -> Any:
    """Configure DSPy with a Groq LM that always uses n=1 at the API."""
    import os

    import dspy
    from dotenv import load_dotenv

    from arcs import config

    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq-backed DSPy optimization")

    lm = GroqSafeLM(
        model=f"openai/{model or config.resolve_generator_model('MEDICAL')}",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=2048,
        n=1,
    )
    dspy.configure(lm=lm)
    return lm


class GroqSafeLM:
    """Factory: returns a ``dspy.LM`` subclass that caps Groq ``n`` at 1."""

    def __new__(cls, **kwargs: Any) -> Any:
        import dspy

        kwargs = dict(kwargs)
        kwargs["n"] = 1
        kwargs.pop("num_generations", None)

        class _GroqSafeLM(dspy.LM):
            """Groq-backed LM: sequential n=1 calls when COPRO requests n>1."""

            def forward(
                self,
                prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> Any:
                requested_n, call_kwargs = clamp_groq_lm_kwargs(kwargs)
                if requested_n <= 1:
                    return super().forward(
                        prompt=prompt, messages=messages, **call_kwargs
                    )

                merged_choices: list[Any] = []
                last: Any = None
                for _ in range(requested_n):
                    last = super().forward(
                        prompt=prompt, messages=messages, **call_kwargs
                    )
                    merged_choices.extend(_response_choices(last))
                return _merge_response_choices(last, merged_choices)

            async def aforward(
                self,
                prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> Any:
                requested_n, call_kwargs = clamp_groq_lm_kwargs(kwargs)
                if requested_n <= 1:
                    return await super().aforward(
                        prompt=prompt, messages=messages, **call_kwargs
                    )

                merged_choices: list[Any] = []
                last: Any = None
                for _ in range(requested_n):
                    last = await super().aforward(
                        prompt=prompt, messages=messages, **call_kwargs
                    )
                    merged_choices.extend(_response_choices(last))
                return _merge_response_choices(last, merged_choices)

        return _GroqSafeLM(**kwargs)


def run_copro(
    student_module: Any,
    trainset: list[Any],
    metric: Callable[..., Any],
    *,
    breadth: int = GROQ_COPRO_DEFAULT_BREADTH,
    depth: int = GROQ_COPRO_DEFAULT_DEPTH,
) -> Any:
    """Compile ``student_module`` with COPRO and return the optimized module."""
    from dspy.teleprompt import COPRO

    validate_copro_breadth(breadth)

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


def ensure_sidecar_written(summary: dict[str, Any], output_path: Path) -> None:
    """Raise if a non-dry-run optimization did not produce a sidecar file."""
    if summary.get("dry_run"):
        return
    if not summary.get("written"):
        raise RuntimeError(
            f"DSPy optimization finished but did not write a sidecar "
            f"(written=false). Check logs above; output={output_path}"
        )
    path = Path(summary.get("output") or output_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Sidecar file missing or empty after optimization: {path}")
