"""Security invariants for generated-code execution."""

from __future__ import annotations

from unittest.mock import MagicMock

from arcs.verification import sandbox


def test_missing_docker_fails_closed_by_default(monkeypatch):
    monkeypatch.delenv("ARCS_ALLOW_UNSAFE_SUBPROCESS", raising=False)
    monkeypatch.setattr(
        sandbox,
        "_run_in_docker",
        MagicMock(side_effect=FileNotFoundError("docker")),
    )
    local = MagicMock(return_value=(0, "", ""))
    monkeypatch.setattr(sandbox, "_run_in_subprocess", local)

    code, _stdout, stderr, issues = sandbox._execute_script("print('safe')")

    assert code == -1
    assert "failed closed" in stderr.lower()
    assert "host fallback is disabled" in stderr.lower()
    assert issues
    local.assert_not_called()


def test_container_program_failure_is_never_retried_on_host(monkeypatch):
    monkeypatch.setenv("ARCS_ALLOW_UNSAFE_SUBPROCESS", "1")
    monkeypatch.setattr(
        sandbox,
        "_run_in_docker",
        MagicMock(return_value=(1, "", "NameError")),
    )
    local = MagicMock(return_value=(0, "", ""))
    monkeypatch.setattr(sandbox, "_run_in_subprocess", local)

    code, _stdout, stderr, issues = sandbox._execute_script("raise NameError")

    assert code == 1
    assert stderr == "NameError"
    assert issues == []
    local.assert_not_called()


def test_unsafe_fallback_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ARCS_ALLOW_UNSAFE_SUBPROCESS", "1")
    monkeypatch.setattr(
        sandbox,
        "_run_in_docker",
        MagicMock(side_effect=FileNotFoundError("docker")),
    )
    local = MagicMock(return_value=(0, "ok", ""))
    monkeypatch.setattr(sandbox, "_run_in_subprocess", local)

    code, stdout, _stderr, issues = sandbox._execute_script("print('dev')")

    assert code == 0
    assert stdout == "ok"
    assert any("UNSAFE" in issue for issue in issues)
    local.assert_called_once()


def test_docker_runner_applies_security_boundaries(monkeypatch):
    completed = MagicMock(returncode=0, stdout="ok", stderr="")
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(sandbox.subprocess, "run", run)

    assert sandbox._run_in_docker("print('ok')") == (0, "ok", "")

    command = run.call_args.args[0]
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop" in command and command[command.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in command
    assert "no-new-privileges" in command
    assert "--user" in command and command[command.index("--user") + 1] == "65534:65534"
    assert "--pids-limit" in command
