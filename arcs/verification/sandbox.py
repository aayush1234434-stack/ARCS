"""
Sandbox verifier — executes Python code and test cases in isolation.

Used only for executable outputs. Does not call LLMs, use specifications,
or make routing decisions.

Usage:
    python sandbox.py --code solution.py --tests tests.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

VERIFICATION_TYPE = "SANDBOX"
DOCKER_IMAGE = os.getenv("SANDBOX_DOCKER_IMAGE", "python:3.11-slim")
TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "10"))

RESULT_TEMPLATE: dict[str, Any] = {
    "verification_type": VERIFICATION_TYPE,
    "verdict": "FAIL",
    "execution_success": False,
    "tests_passed": 0,
    "tests_failed": 0,
    "runtime_error": "",
    "execution_time_ms": 0,
    "issues_found": [],
    "score": 0.0,
    "explanation": "",
}

RESULT_MARKER = "__SANDBOX_RESULT__"


def _normalize_test_cases(test_cases: list) -> list[str]:
    if not isinstance(test_cases, list):
        raise TypeError("test_cases must be a list")

    normalized: list[str] = []
    for index, case in enumerate(test_cases):
        if isinstance(case, str):
            snippet = case.strip()
        elif isinstance(case, dict):
            snippet = str(
                case.get("code") or case.get("assert") or case.get("test") or ""
            ).strip()
        else:
            raise TypeError(f"test_cases[{index}] must be a str or dict")

        if not snippet:
            raise ValueError(f"test_cases[{index}] is empty")
        normalized.append(snippet)

    return normalized


def _build_execution_script(code: str, test_cases: list[str]) -> str:
    """Combine generated code and test cases into one runnable script."""
    tests_literal = json.dumps(test_cases)
    runner = textwrap.dedent(
        f"""
        import json

        _TEST_CASES = json.loads({tests_literal!r})

        _passed = 0
        _failed = 0
        _errors = []

        for _index, _case in enumerate(_TEST_CASES, start=1):
            try:
                exec(_case, globals())
                _passed += 1
            except Exception as _exc:
                _failed += 1
                _errors.append(
                    f"Test {{_index}} failed: {{type(_exc).__name__}}: {{_exc}}"
                )

        print({RESULT_MARKER!r} + json.dumps({{
            "tests_passed": _passed,
            "tests_failed": _failed,
            "errors": _errors,
        }}))
        """
    )
    return textwrap.dedent(f"{code.rstrip()}\n\n{runner}")


def _build_smoke_script(code: str) -> str:
    """Run code only when no explicit test cases are supplied."""
    return textwrap.dedent(
        f"""
        {code.rstrip()}
        print({RESULT_MARKER!r} + '{{"tests_passed": 0, "tests_failed": 0, "errors": []}}')
        """
    )


def _run_in_docker(script: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name

    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "128m",
                "--cpus",
                "0.5",
                "-v",
                f"{script_path}:/script.py:ro",
                DOCKER_IMAGE,
                "python",
                "/script.py",
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return completed.returncode, completed.stdout, completed.stderr
    finally:
        os.unlink(script_path)


def _run_in_subprocess(script: str) -> tuple[int, str, str]:
    """Fallback runner when Docker is unavailable."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name

    try:
        completed = subprocess.run(
            [sys.executable, "-I", script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        return completed.returncode, completed.stdout, completed.stderr
    finally:
        os.unlink(script_path)


def _execute_script(script: str) -> tuple[int, str, str, list[str]]:
    issues: list[str] = []
    docker_error = ""

    try:
        return_code, stdout, stderr = _run_in_docker(script)
        if return_code == 0:
            return return_code, stdout, stderr, issues
        docker_error = stderr.strip() or f"Docker exit code {return_code}"
    except FileNotFoundError:
        docker_error = "Docker executable not found"
    except subprocess.TimeoutExpired:
        return -1, "", f"Execution exceeded {TIMEOUT_SECONDS}s timeout", issues

    issues.append("Docker unavailable or failed; used isolated subprocess fallback.")
    if docker_error:
        issues.append(docker_error[:500])

    return_code, stdout, stderr = _run_in_subprocess(script)
    return return_code, stdout, stderr, issues


def _parse_execution_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            payload = line[len(RESULT_MARKER) :]
            return json.loads(payload)

    raise ValueError("Sandbox did not emit a result marker")


def _build_result(
    *,
    execution_success: bool,
    tests_passed: int,
    tests_failed: int,
    runtime_error: str,
    execution_time_ms: int,
    issues_found: list[str],
    stderr: str = "",
) -> dict[str, Any]:
    result = dict(RESULT_TEMPLATE)
    result["model"] = f"sandbox/{DOCKER_IMAGE}"
    result["execution_success"] = execution_success
    result["tests_passed"] = tests_passed
    result["tests_failed"] = tests_failed
    result["runtime_error"] = runtime_error
    result["execution_time_ms"] = execution_time_ms
    result["issues_found"] = list(issues_found)

    total_tests = tests_passed + tests_failed
    result["score"] = round(tests_passed / total_tests, 4) if total_tests > 0 else 0.0

    if not execution_success:
        result["verdict"] = "FAIL"
        result["explanation"] = runtime_error or "Code execution failed."
    elif total_tests == 0:
        result["verdict"] = "FAIL"
        if "No test cases provided" not in result["issues_found"]:
            result["issues_found"].append("No test cases provided")
        result["explanation"] = (
            "Code executed, but no test cases were provided for verification."
        )
    elif tests_failed > 0:
        result["verdict"] = "FAIL"
        result["explanation"] = f"{tests_failed} of {total_tests} test case(s) failed."
    else:
        result["verdict"] = "PASS"
        result["explanation"] = f"All {tests_passed} test case(s) passed."

    if stderr.strip() and stderr.strip() not in runtime_error:
        result["issues_found"].append(f"stderr: {stderr.strip()[:500]}")

    return result


def run(code: str, test_cases: list) -> dict:
    """
    Execute generated Python code and test cases in an isolated sandbox.

    Args:
        code: Generated Python source code.
        test_cases: List of Python snippets or dicts with a code/assert field.

    Returns:
        Sandbox verification result dict.
    """
    code = code.strip()
    if not code:
        raise ValueError("code cannot be empty")

    normalized_tests = _normalize_test_cases(test_cases) if test_cases else []
    script = (
        _build_execution_script(code, normalized_tests)
        if normalized_tests
        else _build_smoke_script(code)
    )

    started = time.monotonic()
    return_code, stdout, stderr, issues = _execute_script(script)
    execution_time_ms = int((time.monotonic() - started) * 1000)

    if return_code == -1:
        return _build_result(
            execution_success=False,
            tests_passed=0,
            tests_failed=len(normalized_tests),
            runtime_error=stderr,
            execution_time_ms=execution_time_ms,
            issues_found=issues,
            stderr=stderr,
        )

    if return_code != 0:
        runtime_error = stderr.strip() or stdout.strip() or f"Exit code {return_code}"
        return _build_result(
            execution_success=False,
            tests_passed=0,
            tests_failed=len(normalized_tests),
            runtime_error=runtime_error,
            execution_time_ms=execution_time_ms,
            issues_found=issues,
            stderr=stderr,
        )

    try:
        parsed = _parse_execution_output(stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        return _build_result(
            execution_success=False,
            tests_passed=0,
            tests_failed=len(normalized_tests),
            runtime_error=str(exc),
            execution_time_ms=execution_time_ms,
            issues_found=issues + ["Sandbox output could not be parsed."],
            stderr=stderr,
        )

    tests_passed = int(parsed.get("tests_passed", 0))
    tests_failed = int(parsed.get("tests_failed", 0))
    test_errors = [str(item) for item in parsed.get("errors", [])]

    return _build_result(
        execution_success=True,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        runtime_error="",
        execution_time_ms=execution_time_ms,
        issues_found=issues + test_errors,
        stderr=stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sandbox verification on Python code.")
    parser.add_argument("--code", "-c", required=True, help="Path to generated code file")
    parser.add_argument("--tests", "-t", required=True, help="Path to test case JSON list")
    args = parser.parse_args()

    with open(args.code, encoding="utf-8") as handle:
        code = handle.read()
    with open(args.tests, encoding="utf-8") as handle:
        test_cases = json.load(handle)

    try:
        result = run(code, test_cases)
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
