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
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from arcs import progress
from arcs.pipelines import Pipeline, resolve_pipeline
from arcs.pipelines.specialists.common import extract_code_block
from arcs.post import feedback, logger

_components: dict | None = None


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


def _run_judge(
    query: str,
    specialist_result: dict,
    specification: dict,
    components: dict,
) -> dict:
    progress.log("  Calling LLM judge (NVIDIA API)...")
    return components["judge"].run(
        question=query,
        answer=specialist_result.get("answer", ""),
        specification=specification,
    )


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
        test_bundle = tests_future.result()
        first_draft: dict | None = draft_future.result()

    test_cases = test_bundle["test_cases"]
    progress.log(f"  {len(test_cases)} test case(s) from {test_bundle.get('model')}")

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

        code = extract_code_block(specialist_result.get("answer", ""))
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
