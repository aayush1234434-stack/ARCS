"""
ARCS orchestrator — coordinates the pipeline between independent components.

Pipeline:
    query -> router -> specialist -> spec generator -> verifier (sandbox | judge)

This module contains no AI, routing, or verification logic. It only connects
components and collects their outputs into a shared state dictionary.

Usage:
    python main.py "Write a Python function that reverses a string."
    python main.py --quiet "..."   # suppress progress on stderr
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import logger
import progress

FALLBACK_SPECIALIST_NAME = "GENERAL"

_components: dict | None = None


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _load_components() -> dict:
    """Import heavy dependencies lazily so startup progress is visible."""
    progress.log("Loading pipeline modules...")

    progress.log("  judge...")
    import judge

    progress.log("  router...")
    import router

    progress.log("  sandbox...")
    import sandbox

    progress.log("  spec_generator...")
    import spec_generator

    progress.log("  specialists...")
    from specialist import coding, general, legal, medical

    progress.log("Pipeline modules loaded.")

    return {
        "router": router,
        "judge": judge,
        "sandbox": sandbox,
        "spec_generator": spec_generator,
        "specialists": {
            "CODING": coding,
            "MEDICAL": medical,
            "LEGAL": legal,
            "GENERAL": general,
        },
        "fallback": general,
    }


def _get_components() -> dict:
    global _components
    if _components is None:
        _components = _load_components()
    return _components


def _select_specialist(route_result: dict, components: dict):
    """Pick the specialist module based on the router's decision."""
    if route_result.get("use_fallback"):
        return components["fallback"], FALLBACK_SPECIALIST_NAME
    domain = route_result.get("domain", FALLBACK_SPECIALIST_NAME)
    specialist = components["specialists"].get(domain, components["fallback"])
    return specialist, domain


def _extract_code(specialist_result: dict) -> str:
    """Pull runnable code out of the coding specialist's answer."""
    answer = specialist_result.get("answer", "")
    fence = re.search(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL)
    return (fence.group(1) if fence else answer).strip()


def _verify(
    query: str,
    route_result: dict,
    specialist_result: dict,
    specification: dict,
    components: dict,
) -> dict:
    """Dispatch to the sandbox for code, the judge for everything else."""
    if route_result.get("domain") == "CODING" and not route_result.get("use_fallback"):
        code = _extract_code(specialist_result)
        test_cases = specialist_result.get("test_cases", [])
        progress.log(f"  Running sandbox ({len(test_cases)} test case(s))...")
        return components["sandbox"].run(code, test_cases)

    progress.log("  Calling LLM judge (NVIDIA API)...")
    return components["judge"].run(
        question=query,
        answer=specialist_result.get("answer", ""),
        specification=specification,
    )


def run_pipeline(query: str) -> dict:
    """Run one query through the full ARCS pipeline and return the state."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    components = _get_components()

    state = {
        "query": query,
        "route": {},
        "specialist": {},
        "specification": {},
        "verification": {},
    }
    timing: dict[str, int] = {}
    pipeline_start = time.perf_counter()

    with progress.step("Route query (local router)"):
        step_start = time.perf_counter()
        state["route"] = components["router"].route(query)
        timing["route_ms"] = _elapsed_ms(step_start)
        route = state["route"]
        fallback_note = " → using GENERAL fallback" if route.get("use_fallback") else ""
        progress.log(
            f"  Domain: {route.get('domain')} "
            f"(confidence {route.get('confidence', 0):.2f}){fallback_note}"
        )

    specialist_module, specialist_name = _select_specialist(state["route"], components)
    with progress.step(f"Generate answer ({specialist_name} specialist via Groq API)"):
        step_start = time.perf_counter()
        state["specialist"] = specialist_module.run(query)
        timing["specialist_ms"] = _elapsed_ms(step_start)
        answer_preview = state["specialist"].get("answer", "").replace("\n", " ")[:80]
        if answer_preview:
            progress.log(f"  Answer preview: {answer_preview}...")

    with progress.step("Build specification (Qwen via Groq API)"):
        step_start = time.perf_counter()
        state["specification"] = components["spec_generator"].run(query)
        timing["specification_ms"] = _elapsed_ms(step_start)
        required = len(state["specification"].get("required_elements", []))
        progress.log(f"  {required} required element(s) in spec")

    verifier = (
        "sandbox"
        if state["route"].get("domain") == "CODING" and not state["route"].get("use_fallback")
        else "LLM judge"
    )
    with progress.step(f"Verify answer ({verifier})"):
        step_start = time.perf_counter()
        state["verification"] = _verify(
            query=query,
            route_result=state["route"],
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

    logger.log(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
