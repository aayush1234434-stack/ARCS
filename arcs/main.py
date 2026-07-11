"""
ARCS orchestrator — coordinates domain pipelines between independent components.

A specialist pipeline is defined by prompting strategy, structured output
contract, verification mechanism, and optional toolchain. The underlying
language model is interchangeable via config without changing orchestration.

Pipeline:
    query -> router -> resolve pipeline -> specialist -> spec -> verify
    -> feedback (optional) -> log

Usage:
    python main.py "Write a Python function that reverses a string."
    python main.py --quiet "..."
    python main.py --feedback NEGATIVE "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from arcs import progress
from arcs.pipelines import Pipeline, resolve_pipeline
from arcs.pipelines.specialists.common import extract_code_block
from arcs.post import feedback, logger

_components: dict | None = None

ERROR_CLASSES = frozenset(
    {"rate_limit", "judge_parse", "sandbox", "empty_code", "unknown"}
)


class PipelineError(Exception):
    """Pipeline failed; ``state`` holds partial progress, ``query_id``, and ``error_class``."""

    def __init__(self, state: dict[str, Any]):
        self.state = state
        super().__init__(state.get("error") or "pipeline error")


def classify_pipeline_error(exc: BaseException) -> str:
    """Map an exception to a coarse production error bucket."""
    from arcs.clients.rate_limit import is_rate_limit

    if is_rate_limit(exc):
        return "rate_limit"

    message = str(exc).lower()
    module = str(getattr(type(exc), "__module__", "") or "").lower()

    if "judge" in message and ("json" in message or "parse" in message):
        return "judge_parse"
    if "judge response is not" in message or "judge json" in message:
        return "judge_parse"

    if "sandbox" in message or "sandbox" in module:
        return "sandbox"
    if "sandbox did not emit" in message:
        return "sandbox"

    if (
        "no fenced code block" in message
        or "no code block" in message
        or "code cannot be empty" in message
        or "empty code" in message
    ):
        return "empty_code"

    return "unknown"


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _load_components() -> dict:
    """Import heavy dependencies lazily so startup progress is visible."""
    progress.log("Loading pipeline modules...")

    progress.log("  judge...")
    from arcs.verification import judge

    progress.log("  router...")
    from arcs import router

    progress.log("  sandbox...")
    from arcs.verification import sandbox

    progress.log("  spec_generator...")
    from arcs.verification import spec_generator

    progress.log("  test_generator...")
    from arcs.verification import test_generator

    progress.log("  pipelines...")
    from arcs.pipelines import get_registry

    _ = get_registry()

    progress.log("Pipeline modules loaded.")

    return {
        "router": router,
        "judge": judge,
        "sandbox": sandbox,
        "spec_generator": spec_generator,
        "test_generator": test_generator,
    }


def _get_components() -> dict:
    global _components
    if _components is None:
        _components = _load_components()
    return _components


def _sandbox_feedback(verification: dict) -> str:
    """Summarize sandbox failure for specialist retry."""
    parts: list[str] = []
    runtime_error = str(verification.get("runtime_error") or "").strip()
    if runtime_error:
        parts.append(f"Runtime error: {runtime_error}")
    explanation = str(verification.get("explanation") or "").strip()
    if explanation:
        parts.append(explanation)
    issues = verification.get("issues_found") or []
    if isinstance(issues, list) and issues:
        parts.append("Issues:\n- " + "\n- ".join(str(item) for item in issues[:8]))
    if not parts:
        parts.append("Sandbox verification failed.")
    return "\n".join(parts)


def _effective_specialist_answer(specialist_result: dict) -> str:
    """Best-effort full answer text for verification (judge or code extraction).

    The CODING specialist stores code in ``answer`` (SOLUTION section) and prose
    in ``explanation`` / other fields. Prose-only queries (eval-042) may leave
    ``answer`` empty while the explanation holds the real content.
    """
    answer = str(specialist_result.get("answer") or "").strip()
    if answer:
        return answer
    parts: list[str] = []
    for key in ("explanation", "complexity", "edge_cases"):
        value = specialist_result.get(key)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return "\n\n".join(parts)


def _should_defer_to_judge(specialist_result: dict) -> bool:
    """True when the Python sandbox cannot run meaningful verification."""
    text = _effective_specialist_answer(specialist_result)
    if not text:
        return True
    if not _is_python_verifiable(text):
        return True
    return not extract_code_block(text).strip()


def _judge_fallback_tooling(
    test_bundle: dict,
    test_cases: list,
    *,
    model: str,
    pipeline_id: str,
    reason: str,
) -> dict:
    return {
        "test_generator_model": test_bundle.get("model"),
        "test_case_count": len(test_cases),
        "test_cases": test_cases,
        "verifier_fallback": "judge",
        "fallback_reason": reason,
        "generator_model": model,
        "pipeline_id": pipeline_id,
    }


def _empty_answer_verification() -> dict:
    """Synthetic FAIL when the specialist produced no usable answer text."""
    return {
        "verification_type": "LLM_JUDGE",
        "verdict": "FAIL",
        "score": 0.0,
        "missing_required_elements": [],
        "incorrect_claims": [],
        "unsupported_claims": [],
        "disqualifying_conditions_triggered": [],
        "explanation": "specialist returned empty answer",
        "model": "judge/none",
    }


def _run_judge(
    query: str,
    specialist_result: dict,
    specification: dict,
    components: dict,
) -> dict:
    progress.log("  Calling LLM judge (NVIDIA API)...")
    answer = _effective_specialist_answer(specialist_result)
    if not answer.strip():
        return _empty_answer_verification()
    return components["judge"].run(
        question=query,
        answer=answer,
        specification=specification,
    )


_PYTHON_FENCE_LANGS = {"python", "py", "python3"}
_FENCE_LANG_RE = re.compile(r"```([a-zA-Z0-9_+#.-]+)")


def _is_python_verifiable(answer: str) -> bool:
    """Whether the Python sandbox can meaningfully verify this answer.

    The sandbox only runs Python. Answers whose only code fences are another
    language (```javascript, ```sql, ...) or that are pure prose can't be checked
    by it and should fall back to the LLM judge. A ``python`` fence or a bare
    ``` fence (assumed Python, matching extract_code_block) stays on the sandbox.
    """
    if not answer:
        return False
    langs = [lang.lower() for lang in _FENCE_LANG_RE.findall(answer)]
    if any(lang in _PYTHON_FENCE_LANGS for lang in langs):
        return True
    if langs:
        # Only non-Python language fences were tagged.
        return False
    # No language-tagged fence: a bare ``` block is treated as Python; prose is not.
    return "```" in answer


def _empty_code_verification() -> dict:
    """Synthetic sandbox FAIL for answers that contain no runnable code block."""
    return {
        "verification_type": "SANDBOX",
        "verdict": "FAIL",
        "execution_success": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "runtime_error": "No fenced code block found in the answer.",
        "execution_time_ms": 0,
        "issues_found": ["Answer did not contain a ```python code block."],
        "score": 0.0,
        "explanation": "Answer did not contain a runnable code block.",
        "model": "sandbox/none",
    }


def _run_sandbox_pipeline(
    query: str,
    pipeline: Pipeline,
    specification: dict,
    components: dict,
) -> tuple[dict, dict, dict]:
    """
    Coding path: independent tests + generate/verify with retries.

    Returns (specialist_result, verification, tooling_meta).
    """
    model = pipeline.resolve_model()

    # Test generation and the first code draft are independent (both depend only
    # on the query), so run them concurrently to save one LLM round-trip.
    progress.log("  Generating tests + first draft concurrently...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        tests_future = executor.submit(components["test_generator"].run, query)
        draft_future = executor.submit(
            pipeline.specialist.run, query, model=model, feedback=None
        )
        first_draft: dict | None = draft_future.result()
        # A flaky test generator (bad JSON, rate limit, no usable snippets) must
        # not sink the whole coding query. Fall back to a smoke run (no asserts)
        # so the specialist answer is still executed and can PASS on syntax.
        try:
            test_bundle = tests_future.result()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never abort
            progress.log(f"  Test generation failed ({exc}); falling back to smoke run")
            test_bundle = {"test_cases": [], "model": None, "error": str(exc)}

    test_cases = test_bundle["test_cases"]
    progress.log(f"  {len(test_cases)} test case(s) from {test_bundle.get('model')}")

    # Non-Python or prose-only answers can't be verified by the Python sandbox.
    # Defer them to the LLM judge instead of forcing sandbox retries or empty-code runs.
    if first_draft is not None and _should_defer_to_judge(first_draft):
        progress.log("  No runnable Python code — deferring to LLM judge")
        specialist_result = first_draft
        specialist_result["test_cases"] = test_cases
        specialist_result["pipeline_id"] = pipeline.pipeline_id
        specialist_result["generator_model"] = model
        tooling = _judge_fallback_tooling(
            test_bundle,
            test_cases,
            model=model,
            pipeline_id=pipeline.pipeline_id,
            reason="answer is not Python-verifiable (non-Python code, prose-only, or empty code block)",
        )
        return specialist_result, {}, tooling

    specialist_result: dict = {}
    verification: dict = {}
    attempts: list[dict] = []
    feedback_text: str | None = None

    for round_index in range(1, pipeline.max_retries + 1):
        progress.log(
            f"  Coding attempt {round_index}/{pipeline.max_retries} "
            f"(model={model})..."
        )
        if round_index == 1 and first_draft is not None:
            specialist_result = first_draft
        else:
            specialist_result = pipeline.specialist.run(
                query,
                model=model,
                feedback=feedback_text,
            )
        specialist_result["test_cases"] = test_cases
        specialist_result["pipeline_id"] = pipeline.pipeline_id
        specialist_result["generator_model"] = model

        if _should_defer_to_judge(specialist_result):
            progress.log("  No runnable Python code — deferring to LLM judge")
            tooling = _judge_fallback_tooling(
                test_bundle,
                test_cases,
                model=model,
                pipeline_id=pipeline.pipeline_id,
                reason="answer is not Python-verifiable (non-Python code, prose-only, or empty code block)",
            )
            tooling["rounds_used"] = round_index
            return specialist_result, {}, tooling

        code = extract_code_block(_effective_specialist_answer(specialist_result))
        if not code.strip():
            # No fenced code to run. Rather than crash the query, record a FAIL
            # and feed it back so the next attempt emits a real code block.
            progress.log("  No code block in answer — marking FAIL and retrying")
            verification = _empty_code_verification()
        else:
            progress.log(f"  Running sandbox ({len(test_cases)} test case(s))...")
            verification = components["sandbox"].run(code, test_cases)

        attempts.append(
            {
                "round": round_index,
                "verdict": verification.get("verdict"),
                "score": verification.get("score"),
                "issues_found": verification.get("issues_found", []),
            }
        )

        if verification.get("verdict") == "PASS":
            break

        feedback_text = _sandbox_feedback(verification)
        progress.log(f"  Sandbox FAIL — feeding error back to specialist")

    tooling = {
        "test_generator_model": test_bundle.get("model"),
        "test_case_count": len(test_cases),
        "test_cases": test_cases,
        "attempts": attempts,
        "rounds_used": len(attempts),
        "max_retries": pipeline.max_retries,
        "verified": verification.get("verdict") == "PASS",
    }
    if verification.get("verdict") != "PASS":
        tooling["delivery_warning"] = (
            "Code could not be verified after "
            f"{pipeline.max_retries} sandbox attempt(s)."
        )

    # Spec is retained for logging / future hybrid checks; sandbox is authoritative.
    _ = specification
    return specialist_result, verification, tooling


def run_pipeline(query: str) -> dict:
    """Run one query through the full ARCS pipeline and return the state."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    components = _get_components()

    state: dict = {
        "query_id": str(uuid4()),
        "query": query,
        "route": {},
        "pipeline": {},
        "specialist": {},
        "specification": {},
        "verification": {},
        "tooling": {},
    }
    timing: dict[str, int] = {}
    pipeline_start = time.perf_counter()

    try:
        with progress.step("Route query (local router)"):
            step_start = time.perf_counter()
            state["route"] = components["router"].route(query)
            timing["route_ms"] = _elapsed_ms(step_start)
            route = state["route"]
            fallback_note = " → GENERAL fallback" if route.get("use_fallback") else ""
            progress.log(
                f"  Domain: {route.get('domain')} "
                f"(confidence {route.get('confidence', 0):.2f}){fallback_note}"
            )

        pipeline = resolve_pipeline(
            route.get("domain", "GENERAL"),
            use_fallback=bool(route.get("use_fallback")),
        )
        generator_model = pipeline.resolve_model()
        state["pipeline"] = {
            "pipeline_id": pipeline.pipeline_id,
            "verifier": pipeline.verifier,
            "tools": list(pipeline.tools),
            "max_retries": pipeline.max_retries,
            "generator_model": generator_model,
        }
        progress.log(
            f"  Pipeline: {pipeline.pipeline_id} "
            f"(verifier={pipeline.verifier}, model={generator_model})"
        )

        if pipeline.verifier == "sandbox":
            # On the coding path the sandbox is authoritative and the spec is only
            # retained for logging, so generate it concurrently with the sandbox
            # work instead of blocking the answer on it.
            spec_start = time.perf_counter()
            spec_pool = ThreadPoolExecutor(max_workers=1)
            spec_future = spec_pool.submit(components["spec_generator"].run, query)

            with progress.step(
                f"Generate + verify ({pipeline.pipeline_id} sandbox, "
                f"up to {pipeline.max_retries} attempt(s))"
            ):
                step_start = time.perf_counter()
                specialist_result, verification, tooling = _run_sandbox_pipeline(
                    query=query,
                    pipeline=pipeline,
                    specification={},
                    components=components,
                )
                state["specialist"] = specialist_result
                state["verification"] = verification
                state["tooling"] = tooling
                timing["specialist_ms"] = _elapsed_ms(step_start)
                timing["verification_ms"] = timing["specialist_ms"]
                verdict = verification.get("verdict", "UNKNOWN")
                progress.log(
                    f"  Verdict: {verdict} "
                    f"after {tooling.get('rounds_used', 1)} round(s)"
                )
                if tooling.get("delivery_warning"):
                    progress.log(f"  Warning: {tooling['delivery_warning']}")

            try:
                state["specification"] = spec_future.result()
            except Exception as exc:  # spec is non-critical on the coding path
                progress.log(f"  Spec generation failed (non-fatal): {exc}")
                state["specification"] = {}
            finally:
                spec_pool.shutdown(wait=False)
            timing["specification_ms"] = _elapsed_ms(spec_start)

            # Coding answer wasn't Python-verifiable: use the judge (needs the spec).
            if tooling.get("verifier_fallback") == "judge":
                if state["specification"]:
                    with progress.step("Verify (LLM judge — coding fallback)"):
                        judge_start = time.perf_counter()
                        verification = _run_judge(
                            query, specialist_result, state["specification"], components
                        )
                        timing["verification_ms"] = _elapsed_ms(judge_start)
                    state["verification"] = verification
                    state["pipeline"]["verifier"] = "llm_judge"
                    progress.log(
                        f"  Verdict: {verification.get('verdict', 'UNKNOWN')} "
                        f"(judge fallback, score {verification.get('score', 0):.2f})"
                    )
                else:
                    progress.log("  No spec for judge fallback — leaving verdict UNKNOWN")
        else:
            with progress.step("Build specification (Qwen via Groq API)"):
                step_start = time.perf_counter()
                state["specification"] = components["spec_generator"].run(query)
                timing["specification_ms"] = _elapsed_ms(step_start)
                required = len(state["specification"].get("required_elements", []))
                progress.log(f"  {required} required element(s) in spec")

            with progress.step(
                f"Generate answer ({pipeline.pipeline_id} via {generator_model})"
            ):
                step_start = time.perf_counter()
                state["specialist"] = pipeline.specialist.run(query, model=generator_model)
                state["specialist"]["pipeline_id"] = pipeline.pipeline_id
                state["specialist"]["generator_model"] = generator_model
                timing["specialist_ms"] = _elapsed_ms(step_start)
                answer_preview = state["specialist"].get("answer", "").replace("\n", " ")[:80]
                if answer_preview:
                    progress.log(f"  Answer preview: {answer_preview}...")

            with progress.step("Verify answer (LLM judge)"):
                step_start = time.perf_counter()
                state["verification"] = _run_judge(
                    query=query,
                    specialist_result=state["specialist"],
                    specification=state["specification"],
                    components=components,
                )
                timing["verification_ms"] = _elapsed_ms(step_start)
                verdict = state["verification"].get("verdict", "UNKNOWN")
                score = state["verification"].get("score")
                score_note = f", score {score:.2f}" if isinstance(score, (int, float)) else ""
                progress.log(f"  Verdict: {verdict}{score_note}")

        timing["total_ms"] = _elapsed_ms(pipeline_start)
        state["timing"] = timing

        progress.log("Pipeline complete.")
        return state
    except Exception as exc:
        state["error"] = str(exc)
        state["error_class"] = classify_pipeline_error(exc)
        timing["total_ms"] = _elapsed_ms(pipeline_start)
        state["timing"] = timing
        raise PipelineError(state) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a query through the ARCS pipeline.")
    parser.add_argument("query", nargs="?", help="User query (prompted if omitted)")
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages (JSON output only)",
    )
    parser.add_argument(
        "--feedback",
        choices=("POSITIVE", "NEGATIVE"),
        help="Explicit user feedback signal (skips interactive prompt)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip interactive feedback prompt after the answer is delivered",
    )
    args = parser.parse_args()

    progress.set_verbose(not args.quiet)

    query = args.query
    if not query:
        query = input("Enter your query: ").strip()
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            sys.exit(1)

    progress.log(f"Query: {query[:120]}{'...' if len(query) > 120 else ''}")

    try:
        state = run_pipeline(query)
    except PipelineError as exc:
        progress.log(
            f"Pipeline error (query_id={exc.state.get('query_id')}): {exc}"
        )
        logger.log(exc.state)
        raise SystemExit(1) from exc
    except Exception as exc:
        progress.log(f"Pipeline error: {exc}")
        raise

    with progress.step("Collect user feedback"):
        feedback_signal = feedback.collect(
            explicit=args.feedback,
            interactive=not args.quiet and not args.no_feedback and args.feedback is None,
        )
        if feedback_signal:
            progress.log(f"  Signal: {feedback_signal['user_feedback']} ({feedback_signal['source']})")
        else:
            progress.log("  No feedback collected")

    log_record = feedback.apply(state, feedback_signal)
    logger.log(log_record)
    print(json.dumps(log_record, indent=2))


if __name__ == "__main__":
    main()
