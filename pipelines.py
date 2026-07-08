"""
Pipeline registry — domain-specific processing contracts for ARCS.

A pipeline is defined by:
  - prompting strategy (specialist module)
  - structured output contract (parser in specialist module)
  - verification mechanism (sandbox | judge)
  - optional toolchain (e.g. independent test generation + retries)

The underlying language model is resolved from config and is interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import config

VerifierKind = Literal["sandbox", "judge"]


@dataclass(frozen=True)
class Pipeline:
    """Immutable description of one domain processing path."""

    domain: str
    verifier: VerifierKind
    specialist: Any  # module with run(query, *, model=..., feedback=None) -> dict
    max_retries: int = 1
    tools: tuple[str, ...] = field(default_factory=tuple)
    requires_spec: bool = True

    @property
    def pipeline_id(self) -> str:
        return self.domain

    def resolve_model(self) -> str:
        return config.resolve_generator_model(self.domain)


def _load_registry() -> dict[str, Pipeline]:
    from specialist import coding, general, legal, medical

    return {
        "CODING": Pipeline(
            domain="CODING",
            verifier="sandbox",
            specialist=coding,
            max_retries=config.CODING_MAX_RETRIES,
            tools=("test_generator", "sandbox_retry"),
            requires_spec=True,
        ),
        "MEDICAL": Pipeline(
            domain="MEDICAL",
            verifier="judge",
            specialist=medical,
            max_retries=1,
            tools=(),
            requires_spec=True,
        ),
        "LEGAL": Pipeline(
            domain="LEGAL",
            verifier="judge",
            specialist=legal,
            max_retries=1,
            tools=(),
            requires_spec=True,
        ),
        "GENERAL": Pipeline(
            domain="GENERAL",
            verifier="judge",
            specialist=general,
            max_retries=1,
            tools=(),
            requires_spec=True,
        ),
    }


_REGISTRY: dict[str, Pipeline] | None = None


def get_registry() -> dict[str, Pipeline]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_registry()
    return _REGISTRY


def resolve_pipeline(domain: str, *, use_fallback: bool = False) -> Pipeline:
    """Select the pipeline for a routed domain, with GENERAL as fallback."""
    registry = get_registry()
    if use_fallback:
        return registry["GENERAL"]
    return registry.get(domain.upper(), registry["GENERAL"])


def list_pipelines() -> list[str]:
    return sorted(get_registry().keys())
