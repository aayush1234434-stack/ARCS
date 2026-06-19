"""
verifier/sandbox.py  —  Docker-based execution environment for code answers.

Runs the specialist's code against its own test cases in an isolated container.
Retries up to MAX_ROUNDS on failure, passing the error back to the specialist
each round so it can self-correct.

Requires Docker to be running locally.
"""

import subprocess
import tempfile
import textwrap
import os

MAX_ROUNDS = 3
TIMEOUT_SECONDS = 10

# Docker image to use for execution.
# python:3.11-slim is small, has no network, and is destroyed after each run.
DOCKER_IMAGE = "python:3.11-slim"


def _build_execution_script(code: str, test_cases: str) -> str:
    """
    Combine specialist code and test cases into a single executable script.
    Wraps test cases in a try/except so failures produce readable output.
    """
    return textwrap.dedent(f"""\
        # --- specialist code ---
        {code}

        # --- test cases ---
        try:
            {textwrap.indent(test_cases, '    ')}
            print("__ARCS_PASS__")
        except Exception as e:
            print(f"__ARCS_FAIL__: {{type(e).__name__}}: {{e}}")
    """)


def _run_in_docker(script: str) -> tuple[bool, str]:
    """
    Write script to a temp file, mount it into a fresh Docker container,
    run it with a strict timeout, and return (passed, output).

    The container has:
      - no network access (--network none)
      - no access to the host filesystem beyond the single script file
      - a hard timeout enforced by subprocess
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "128m",
                "--cpus",   "0.5",
                "-v", f"{script_path}:/script.py:ro",
                DOCKER_IMAGE,
                "python", "/script.py",
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        output = result.stdout.strip() + result.stderr.strip()
        passed = "__ARCS_PASS__" in output
        return passed, output

    except subprocess.TimeoutExpired:
        return False, f"__ARCS_FAIL__: TimeoutError: execution exceeded {TIMEOUT_SECONDS}s"

    except FileNotFoundError:
        # Docker is not installed or not running
        return False, "__ARCS_FAIL__: DockerNotFound: Docker is not available on this machine"

    finally:
        os.unlink(script_path)


def run(code: str, test_cases: str) -> dict:
    """
    Run code against test cases with up to MAX_ROUNDS retry attempts.

    On each failure the error is returned so the orchestrator can pass it
    back to the coding specialist for self-correction before the next round.

    Args:
        code:        Raw Python code string from the coding specialist.
        test_cases:  Raw Python test case string from the coding specialist.

    Returns:
        {
            result:       "PASS" | "FAIL",
            rounds:       int,
            final_error:  str | None,   # last error message if all rounds failed
            round_errors: list[str],    # error from each failed round
        }
    """
    round_errors: list[str] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        script = _build_execution_script(code, test_cases)
        passed, output = _run_in_docker(script)

        if passed:
            return {
                "result":       "PASS",
                "rounds":       round_num,
                "final_error":  None,
                "round_errors": round_errors,
            }

        # Extract the error message for the retry loop
        error_line = next(
            (l for l in output.splitlines() if "__ARCS_FAIL__" in l),
            output
        )
        error_msg = error_line.replace("__ARCS_FAIL__: ", "").strip()
        round_errors.append(f"Round {round_num}: {error_msg}")

        # If not the last round, caller (orchestrator) should send
        # error_msg back to the specialist and call run() again with
        # the corrected code.
        if round_num < MAX_ROUNDS:
            return {
                "result":       "FAIL",
                "rounds":       round_num,
                "final_error":  error_msg,
                "round_errors": round_errors,
                "needs_retry":  True,   # signal to orchestrator: send error back to specialist
            }

    # All rounds exhausted
    return {
        "result":       "FAIL",
        "rounds":       MAX_ROUNDS,
        "final_error":  round_errors[-1] if round_errors else "Unknown error",
        "round_errors": round_errors,
        "needs_retry":  False,
    }