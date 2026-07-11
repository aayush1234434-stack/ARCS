"""
DSPy module for optimizing the GENERAL specialist system prompt.

Uses COPRO to rewrite signature instructions (the system prompt). Optimized
text is written to ``artifacts/prompts/general_optimized.txt`` — never
auto-applied to ``general.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from arcs import config
from arcs.optimization.dspy_common import (
    GROQ_COPRO_DEFAULT_BREADTH,
    GROQ_COPRO_DEFAULT_DEPTH,
    configure_groq_lm,
    extract_instructions,
    run_copro,
    save_sidecar_prompt,
)
from arcs.optimization.metrics import judge_metric
from arcs.pipelines.specialists.general import SYSTEM_PROMPT as GENERAL_SYSTEM_PROMPT

DEFAULT_QUEUE = config.LOGS_DIR / "queues" / "specialist_queue.jsonl"
DEFAULT_OUTPUT = config.ARTIFACTS_DIR / "prompts" / "general_optimized.txt"
VALID_DOMAINS = frozenset({"GENERAL"})


def configure_lm(*, model: str | None = None) -> Any:
    """Configure DSPy to use Groq via the OpenAI-compatible API."""
    return configure_groq_lm(model or config.resolve_generator_model("GENERAL"))


def _build_signature(instructions: str):
    import dspy

    class GeneralAnswerSignature(dspy.Signature):
        """Produce a structured general specialist answer."""

        query: str = dspy.InputField(desc="User general or cross-domain question")
        answer: str = dspy.OutputField(
            desc=(
                "Structured general answer with ANSWER / KEY CLAIMS / "
                "CAVEATS / UNCERTAINTY sections"
            )
        )

    return GeneralAnswerSignature.with_instructions(instructions)


def build_general_module(instructions: str | None = None):
    """Return a dspy.Module that generates general answers from a query."""
    import dspy

    prompt = instructions or GENERAL_SYSTEM_PROMPT
    signature = _build_signature(prompt)

    class GeneralSpecialist(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(signature)

        def forward(self, query: str):
            return self.generate(query=query)

    return GeneralSpecialist()


def _pipeline_id(record: dict[str, Any]) -> str | None:
    pipeline = record.get("pipeline")
    if isinstance(pipeline, dict) and pipeline.get("pipeline_id"):
        return str(pipeline["pipeline_id"]).upper()
    specialist = record.get("specialist")
    if isinstance(specialist, dict) and specialist.get("pipeline_id"):
        return str(specialist["pipeline_id"]).upper()
    if isinstance(specialist, dict) and specialist.get("domain"):
        return str(specialist["domain"]).upper()
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("pipeline_id"):
        return str(metadata["pipeline_id"]).upper()
    return None


def load_general_examples(
    queue_path: Path,
    *,
    max_examples: int = 20,
) -> list[Any]:
    """Load GENERAL specialist-queue rows as dspy.Example objects."""
    import dspy

    if not queue_path.exists():
        raise FileNotFoundError(
            f"specialist queue not found: {queue_path}\n"
            "Run: python scripts/extract_queues.py"
        )

    examples: list[Any] = []
    skipped = 0
    with queue_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                continue

            pipeline_id = _pipeline_id(record)
            if pipeline_id is None or pipeline_id not in VALID_DOMAINS:
                skipped += 1
                continue

            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                skipped += 1
                continue

            specification = record.get("specification")
            if not isinstance(specification, dict) or not specification:
                print(
                    f"Warning: skipping record without specification "
                    f"(query_id={record.get('query_id')!r})",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            specialist = record.get("specialist") or {}
            prior_answer = ""
            if isinstance(specialist, dict):
                prior_answer = str(specialist.get("answer") or "")

            example = dspy.Example(
                query=query.strip(),
                question=query.strip(),
                specification=specification,
                spec=specification,
                prior_answer=prior_answer,
                user_feedback=record.get("user_feedback") or "NEGATIVE",
                query_id=record.get("query_id"),
            ).with_inputs("query")
            examples.append(example)

            if len(examples) >= max_examples:
                break

    if skipped:
        print(f"Skipped {skipped} non-GENERAL or incomplete row(s).", file=sys.stderr)
    return examples


def extract_optimized_instructions(module: Any) -> str:
    """Pull the optimized signature instructions from a DSPy module."""
    return extract_instructions(module, GENERAL_SYSTEM_PROMPT)


def save_optimized_prompt(text: str, output_path: Path) -> Path:
    """Write sidecar prompt file (does not modify general.py)."""
    return save_sidecar_prompt(
        text,
        output_path,
        component_name="GENERAL specialist",
        source_file="arcs/pipelines/specialists/general.py",
    )


def optimize_general_prompt(
    *,
    queue_path: Path | None = None,
    output_path: Path | None = None,
    max_examples: int = 20,
    dry_run: bool = False,
    breadth: int = GROQ_COPRO_DEFAULT_BREADTH,
    depth: int = GROQ_COPRO_DEFAULT_DEPTH,
) -> dict[str, Any]:
    """Run COPRO on GENERAL specialist failures and write a sidecar prompt.

    Returns a summary dict with counts and output path.
    """
    source = queue_path or DEFAULT_QUEUE
    destination = output_path or DEFAULT_OUTPUT

    examples = load_general_examples(source, max_examples=max_examples)
    summary: dict[str, Any] = {
        "examples": len(examples),
        "queue": str(source),
        "output": str(destination),
        "dry_run": dry_run,
        "written": False,
    }

    if not examples:
        raise ValueError(
            f"No GENERAL examples in {source}. "
            "Collect NEGATIVE SPECIALIST failures for GENERAL queries first "
            "(demo/batch run + python scripts/extract_queues.py), then retry."
        )

    if dry_run:
        print(
            f"Dry-run: would optimize on {len(examples)} GENERAL example(s) "
            f"from {source}",
            file=sys.stderr,
        )
        for index, example in enumerate(examples, start=1):
            query = getattr(example, "query", "")
            preview = query.replace("\n", " ")[:72]
            print(f"  [{index}] {preview}", file=sys.stderr)
        return summary

    configure_lm()

    # Hold out ~20% when we have enough examples; else train on all.
    if len(examples) >= 5:
        split = max(1, len(examples) // 5)
        trainset = examples[:-split]
        valset = examples[-split:]
    else:
        trainset = examples
        valset = examples

    student = build_general_module()

    print(
        f"Optimizing GENERAL prompt on {len(trainset)} train / "
        f"{len(valset)} val example(s) (COPRO breadth={breadth}, depth={depth})...",
        file=sys.stderr,
    )
    optimized = run_copro(
        student,
        trainset,
        judge_metric,
        breadth=breadth,
        depth=depth,
    )

    prompt_text = extract_optimized_instructions(optimized)
    save_optimized_prompt(prompt_text, destination)
    summary["written"] = True
    summary["prompt_chars"] = len(prompt_text)
    print(f"Wrote optimized prompt → {destination}", file=sys.stderr)
    print(
        "Review the file, then manually update SYSTEM_PROMPT in "
        "arcs/pipelines/specialists/general.py if approved.",
        file=sys.stderr,
    )
    return summary
