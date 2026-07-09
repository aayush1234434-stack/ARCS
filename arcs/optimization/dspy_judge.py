"""
DSPy module for optimizing the LLM judge (verifier) system prompt.

Uses COPRO on VERIFIER-attributed failures where the judge wrongly passed a
bad answer (high score + NEGATIVE user feedback). Writes a sidecar prompt to
``artifacts/prompts/judge_optimized.txt`` — never auto-edits ``judge.py``.

Sandbox path is out of scope; LLM judge only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from arcs import config
from arcs.optimization.metrics import verifier_false_pass_metric
from arcs.verification.judge import (
    SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT,
    _build_user_message,
    _extract_json,
    _normalize_result,
    _validate_specification,
)

DEFAULT_QUEUE = config.LOGS_DIR / "queues" / "verifier_queue.jsonl"
DEFAULT_OUTPUT = config.ARTIFACTS_DIR / "prompts" / "judge_optimized.txt"
HIGH_SCORE_THRESHOLD = 0.80


def configure_lm(*, model: str | None = None) -> Any:
    """Configure DSPy to use the NVIDIA OpenAI-compatible judge endpoint."""
    import dspy
    from dotenv import load_dotenv

    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for judge prompt optimization")

    lm = dspy.LM(
        model=f"openai/{model or config.DEFAULT_JUDGE_MODEL}",
        api_key=api_key,
        api_base="https://integrate.api.nvidia.com/v1",
        temperature=0.1,
        max_tokens=1024,
    )
    dspy.configure(lm=lm)
    return lm


def _build_signature(instructions: str):
    import dspy

    class JudgeVerificationSignature(dspy.Signature):
        """Compare an answer against a specification checklist."""

        question: str = dspy.InputField(desc="Original user question")
        answer: str = dspy.InputField(desc="Specialist answer text")
        specification_json: str = dspy.InputField(
            desc="Specification checklist as JSON string"
        )
        verification_json: str = dspy.OutputField(
            desc=(
                "ONLY valid JSON with keys: verification_type, verdict, score, "
                "missing_required_elements, incorrect_claims, unsupported_claims, "
                "disqualifying_conditions_triggered, explanation"
            )
        )

    return JudgeVerificationSignature.with_instructions(instructions)


def build_judge_module(instructions: str | None = None):
    """Return a dspy.Module that emits judge JSON for (question, answer, spec)."""
    import dspy

    prompt = instructions or JUDGE_SYSTEM_PROMPT
    signature = _build_signature(prompt)

    class JudgeVerifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(signature)

        def forward(self, question: str, answer: str, specification_json: str):
            return self.generate(
                question=question,
                answer=answer,
                specification_json=specification_json,
            )

    return JudgeVerifier()


def _is_llm_judge_record(record: dict[str, Any]) -> bool:
    verification = record.get("verification") or {}
    if not isinstance(verification, dict):
        return False
    vtype = str(verification.get("verification_type") or "").upper()
    if vtype == "SANDBOX":
        return False
    # Prefer explicit LLM_JUDGE; also accept missing type if score/verdict present
    if vtype and vtype not in {"LLM_JUDGE", "JUDGE"}:
        return False
    return "verdict" in verification or "score" in verification or vtype == "LLM_JUDGE"


def _was_false_pass(record: dict[str, Any]) -> bool:
    """True when original verifier approved an answer the user rejected."""
    feedback = record.get("user_feedback")
    if feedback is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            feedback = metadata.get("user_feedback")
    if str(feedback or "").strip().upper() != "NEGATIVE":
        return False

    verification = record.get("verification") or {}
    if not isinstance(verification, dict):
        return False

    verdict = str(verification.get("verdict") or "").strip().upper()
    try:
        score = float(verification.get("score")) if verification.get("score") is not None else None
    except (TypeError, ValueError):
        score = None

    if verdict == "PASS":
        return True
    if score is not None and score >= HIGH_SCORE_THRESHOLD:
        return True
    return False


def load_verifier_examples(
    queue_path: Path,
    *,
    max_examples: int = 20,
) -> list[Any]:
    """Load verifier-queue false-PASS rows as dspy.Example objects."""
    import dspy

    if not queue_path.exists():
        raise FileNotFoundError(
            f"verifier queue not found: {queue_path}\n"
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

            if not _is_llm_judge_record(record):
                skipped += 1
                continue
            if not _was_false_pass(record):
                skipped += 1
                continue

            question = record.get("query")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue

            specialist = record.get("specialist") or {}
            answer = ""
            if isinstance(specialist, dict):
                answer = str(specialist.get("answer") or "")
            if not answer.strip():
                answer = str(record.get("response") or "")
            if not answer.strip():
                print(
                    f"Warning: skipping record without answer "
                    f"(query_id={record.get('query_id')!r})",
                    file=sys.stderr,
                )
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

            try:
                _validate_specification(specification)
            except (TypeError, ValueError) as exc:
                print(
                    f"Warning: skipping invalid specification "
                    f"(query_id={record.get('query_id')!r}): {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            verification = record.get("verification") or {}
            original_score = verification.get("score") if isinstance(verification, dict) else None
            original_verdict = (
                verification.get("verdict") if isinstance(verification, dict) else None
            )

            # Pack spec as JSON string for the signature input field.
            spec_json = json.dumps(
                {key: specification[key] for key in (
                    "intent",
                    "required_elements",
                    "correctness_criteria",
                    "disqualifying_conditions",
                    "scope",
                ) if key in specification},
                ensure_ascii=False,
            )

            example = dspy.Example(
                question=question.strip(),
                answer=answer.strip(),
                specification=specification,
                specification_json=spec_json,
                user_feedback="NEGATIVE",
                expected_verdict="FAIL",
                target_verdict="FAIL",
                original_verdict=original_verdict,
                original_score=original_score,
                query_id=record.get("query_id"),
                # Keep a packed user message for debugging / reuse of judge helpers.
                judge_user_message=_build_user_message(
                    question.strip(), answer.strip(), specification
                ),
            ).with_inputs("question", "answer", "specification_json")
            examples.append(example)

            if len(examples) >= max_examples:
                break

    if skipped:
        print(
            f"Skipped {skipped} non-false-PASS / sandbox / incomplete row(s).",
            file=sys.stderr,
        )
    return examples


def extract_optimized_instructions(module: Any) -> str:
    """Pull optimized signature instructions from a DSPy judge module."""
    generate = getattr(module, "generate", None)
    if generate is None:
        return JUDGE_SYSTEM_PROMPT

    signature = getattr(generate, "signature", None)
    if signature is None:
        return JUDGE_SYSTEM_PROMPT

    instructions = getattr(signature, "instructions", None)
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    return JUDGE_SYSTEM_PROMPT


def save_optimized_prompt(text: str, output_path: Path) -> Path:
    """Write sidecar prompt file (does not modify judge.py)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Optimized LLM judge system prompt (DSPy COPRO)\n"
        "# Review this file, then manually replace SYSTEM_PROMPT in\n"
        "# arcs/verification/judge.py if it looks better.\n"
        "# Do not auto-apply without human review.\n"
        "# " + ("-" * 60) + "\n\n"
    )
    output_path.write_text(header + text.strip() + "\n", encoding="utf-8")
    return output_path


def parse_judge_prediction(raw: str) -> dict[str, Any]:
    """Reuse judge parsing helpers on a raw model string."""
    parsed = _extract_json(raw)
    return _normalize_result(parsed)


def optimize_judge_prompt(
    *,
    queue_path: Path | None = None,
    output_path: Path | None = None,
    max_examples: int = 20,
    dry_run: bool = False,
    breadth: int = 5,
    depth: int = 2,
) -> dict[str, Any]:
    """Run COPRO on verifier false-PASS failures and write a sidecar prompt."""
    source = queue_path or DEFAULT_QUEUE
    destination = output_path or DEFAULT_OUTPUT

    examples = load_verifier_examples(source, max_examples=max_examples)
    summary: dict[str, Any] = {
        "examples": len(examples),
        "queue": str(source),
        "output": str(destination),
        "dry_run": dry_run,
        "written": False,
    }

    if not examples:
        raise ValueError(
            f"No LLM-judge false-PASS examples in {source}. "
            f"Need VERIFIER-attributed rows where the judge passed "
            f"(or scored >= {HIGH_SCORE_THRESHOLD}) but user said NEGATIVE. "
            "Collect feedback + run extract_queues first."
        )

    if dry_run:
        print(
            f"Dry-run: would optimize on {len(examples)} verifier false-PASS "
            f"example(s) from {source}",
            file=sys.stderr,
        )
        for index, example in enumerate(examples, start=1):
            question = getattr(example, "question", "")
            preview = question.replace("\n", " ")[:72]
            orig = getattr(example, "original_verdict", "?")
            score = getattr(example, "original_score", "?")
            print(
                f"  [{index}] orig={orig}/{score}  {preview}",
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

    from dspy.teleprompt import COPRO

    student = build_judge_module()
    optimizer = COPRO(
        metric=verifier_false_pass_metric,
        breadth=breadth,
        depth=depth,
        init_temperature=1.0,
        track_stats=True,
    )

    print(
        f"Optimizing judge prompt on {len(trainset)} train / "
        f"{len(valset)} val example(s) (COPRO breadth={breadth}, depth={depth})...",
        file=sys.stderr,
    )
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        eval_kwargs={"num_threads": 1},
    )

    prompt_text = extract_optimized_instructions(optimized)
    save_optimized_prompt(prompt_text, destination)
    summary["written"] = True
    summary["prompt_chars"] = len(prompt_text)
    print(f"Wrote optimized prompt → {destination}", file=sys.stderr)
    print(
        "Review the file, then manually update SYSTEM_PROMPT in "
        "arcs/verification/judge.py if approved.",
        file=sys.stderr,
    )
    return summary
