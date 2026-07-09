"""
Experimental DSPy module for optimizing the specification-generator prompt.

Only loads queue rows that look like incomplete specs (few required_elements
or non-empty judge missing_required_elements). Writes a sidecar to
``artifacts/prompts/spec_optimized.txt`` — never auto-applies to
``spec_generator.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from arcs import config
from arcs.optimization.dspy_common import (
    configure_groq_lm,
    extract_instructions,
    run_copro,
    save_sidecar_prompt,
)
from arcs.optimization.metrics import spec_metric
from arcs.verification.spec_generator import SYSTEM_PROMPT as SPEC_SYSTEM_PROMPT

DEFAULT_SPECIALIST_QUEUE = config.LOGS_DIR / "queues" / "specialist_queue.jsonl"
DEFAULT_VERIFIER_QUEUE = config.LOGS_DIR / "queues" / "verifier_queue.jsonl"
DEFAULT_OUTPUT = config.ARTIFACTS_DIR / "prompts" / "spec_optimized.txt"

# Specs with fewer than this many required_elements are treated as incomplete.
MIN_REQUIRED_ELEMENTS = 3


def configure_lm(*, model: str | None = None) -> Any:
    """Configure DSPy to use Groq (same family as the live spec generator)."""
    return configure_groq_lm(model or config.DEFAULT_SPEC_MODEL)


def _build_signature(instructions: str):
    import dspy

    class SpecGeneratorSignature(dspy.Signature):
        """Produce a structured answer specification for a user query."""

        query: str = dspy.InputField(desc="User question to specify")
        specification: str = dspy.OutputField(
            desc=(
                "Structured specification with INTENT / REQUIRED ELEMENTS / "
                "CORRECTNESS CRITERIA / DISQUALIFYING CONDITIONS / SCOPE "
                "section headers (same format as the live spec generator)"
            )
        )

    return SpecGeneratorSignature.with_instructions(instructions)


def build_spec_module(instructions: str | None = None):
    """Return a dspy.Module that generates specifications from a query."""
    import dspy

    prompt = instructions or SPEC_SYSTEM_PROMPT
    signature = _build_signature(prompt)

    class SpecGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(signature)

        def forward(self, query: str):
            return self.generate(query=query)

    return SpecGenerator()


def _required_count(specification: Any) -> int:
    if not isinstance(specification, dict):
        return 0
    required = specification.get("required_elements") or []
    if not isinstance(required, list):
        return 0
    return len([item for item in required if str(item).strip()])


def _missing_required(record: dict[str, Any]) -> list[str]:
    verification = record.get("verification") or {}
    if not isinstance(verification, dict):
        return []
    missing = verification.get("missing_required_elements") or []
    if not isinstance(missing, list):
        return []
    return [str(item).strip() for item in missing if str(item).strip()]


def _is_incomplete_spec(record: dict[str, Any]) -> bool:
    """True when the logged spec looks thin or the judge flagged missing elements."""
    specification = record.get("specification")
    if not isinstance(specification, dict) or not specification:
        return False
    if _required_count(specification) < MIN_REQUIRED_ELEMENTS:
        return True
    if _missing_required(record):
        return True
    return False


def _extract_answer(record: dict[str, Any]) -> str:
    specialist = record.get("specialist") or {}
    if isinstance(specialist, dict):
        answer = str(specialist.get("answer") or "").strip()
        if answer:
            return answer
    return str(record.get("response") or "").strip()


def _baseline_judge_score(record: dict[str, Any]) -> float:
    verification = record.get("verification") or {}
    if isinstance(verification, dict) and verification.get("score") is not None:
        try:
            return float(verification["score"])
        except (TypeError, ValueError):
            pass
    return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number} "
                    f"of {path}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def load_incomplete_spec_examples(
    *,
    specialist_queue: Path | None = None,
    verifier_queue: Path | None = None,
    max_examples: int = 20,
) -> list[Any]:
    """Load incomplete-spec rows from specialist and verifier queues."""
    import dspy

    specialist_path = specialist_queue or DEFAULT_SPECIALIST_QUEUE
    verifier_path = verifier_queue or DEFAULT_VERIFIER_QUEUE

    if not specialist_path.exists() and not verifier_path.exists():
        raise FileNotFoundError(
            "Neither specialist nor verifier queue found.\n"
            f"  looked for: {specialist_path}\n"
            f"  looked for: {verifier_path}\n"
            "Run: python scripts/extract_queues.py"
        )

    examples: list[Any] = []
    skipped = 0
    seen_queries: set[str] = set()

    for path in (specialist_path, verifier_path):
        for record in _read_jsonl(path):
            if not _is_incomplete_spec(record):
                skipped += 1
                continue

            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                skipped += 1
                continue

            answer = _extract_answer(record)
            if not answer:
                print(
                    f"Warning: skipping incomplete-spec row without answer "
                    f"(query_id={record.get('query_id')!r})",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            key = query.strip().lower()
            if key in seen_queries:
                skipped += 1
                continue
            seen_queries.add(key)

            specification = record.get("specification") or {}
            baseline_count = _required_count(specification)
            baseline_score = _baseline_judge_score(record)
            missing = _missing_required(record)

            example = dspy.Example(
                query=query.strip(),
                question=query.strip(),
                answer=answer,
                prior_answer=answer,
                specification=specification,
                spec=specification,
                baseline_required_count=baseline_count,
                required_count=baseline_count,
                baseline_judge_score=baseline_score,
                prior_judge_score=baseline_score,
                missing_required_elements=missing,
                user_feedback=record.get("user_feedback") or "NEGATIVE",
                query_id=record.get("query_id"),
                source_queue=str(path),
            ).with_inputs("query")
            examples.append(example)

            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    if skipped:
        print(
            f"Skipped {skipped} complete-spec / incomplete / duplicate row(s).",
            file=sys.stderr,
        )
    return examples


def extract_optimized_instructions(module: Any) -> str:
    """Pull optimized signature instructions from a DSPy spec module."""
    return extract_instructions(module, SPEC_SYSTEM_PROMPT)


def save_optimized_prompt(text: str, output_path: Path) -> Path:
    """Write sidecar prompt file (does not modify spec_generator.py)."""
    return save_sidecar_prompt(
        text,
        output_path,
        component_name="spec generator (experimental)",
        source_file="arcs/verification/spec_generator.py",
    )


def optimize_spec_prompt(
    *,
    specialist_queue: Path | None = None,
    verifier_queue: Path | None = None,
    output_path: Path | None = None,
    max_examples: int = 20,
    dry_run: bool = False,
    breadth: int = 5,
    depth: int = 2,
) -> dict[str, Any]:
    """Run COPRO on incomplete-spec failures and write a sidecar prompt.

    Returns a summary dict. Raises ValueError when no incomplete-spec examples
    are found (empty queues or all specs look complete).
    """
    destination = output_path or DEFAULT_OUTPUT
    specialist_path = specialist_queue or DEFAULT_SPECIALIST_QUEUE
    verifier_path = verifier_queue or DEFAULT_VERIFIER_QUEUE

    examples = load_incomplete_spec_examples(
        specialist_queue=specialist_path,
        verifier_queue=verifier_path,
        max_examples=max_examples,
    )
    summary: dict[str, Any] = {
        "examples": len(examples),
        "specialist_queue": str(specialist_path),
        "verifier_queue": str(verifier_path),
        "output": str(destination),
        "dry_run": dry_run,
        "written": False,
        "experimental": True,
        "min_required_elements": MIN_REQUIRED_ELEMENTS,
    }

    if not examples:
        raise ValueError(
            "No incomplete-spec examples found in specialist/verifier queues. "
            f"(Need required_elements < {MIN_REQUIRED_ELEMENTS} or non-empty "
            "verification.missing_required_elements.) "
            "Collect NEGATIVE feedback + run extract_queues, or skip this "
            "experimental optimizer when specs look complete."
        )

    if dry_run:
        print(
            f"Dry-run (experimental): would optimize on {len(examples)} "
            f"incomplete-spec example(s)",
            file=sys.stderr,
        )
        for index, example in enumerate(examples, start=1):
            query = getattr(example, "query", "")
            preview = query.replace("\n", " ")[:64]
            n_req = getattr(example, "baseline_required_count", 0)
            n_miss = len(getattr(example, "missing_required_elements", None) or [])
            print(
                f"  [{index}] req={n_req} missing={n_miss}  {preview}",
                file=sys.stderr,
            )
        return summary

    configure_lm()

    if len(examples) >= 5:
        split = max(1, len(examples) // 5)
        trainset = examples[:-split]
        valset = examples[-split:]
    else:
        trainset = examples
        valset = examples

    student = build_spec_module()

    print(
        f"Optimizing spec-generator prompt on {len(trainset)} train / "
        f"{len(valset)} val example(s) (COPRO breadth={breadth}, depth={depth}) "
        "(experimental)...",
        file=sys.stderr,
    )
    optimized = run_copro(
        student,
        trainset,
        spec_metric,
        breadth=breadth,
        depth=depth,
    )

    prompt_text = extract_optimized_instructions(optimized)
    save_optimized_prompt(prompt_text, destination)
    summary["written"] = True
    summary["prompt_chars"] = len(prompt_text)
    print(f"Wrote optimized prompt → {destination}", file=sys.stderr)
    print(
        "EXPERIMENTAL: review the file, then manually update SYSTEM_PROMPT in "
        "arcs/verification/spec_generator.py if approved.",
        file=sys.stderr,
    )
    return summary
