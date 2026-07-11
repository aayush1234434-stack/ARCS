"""apply_sidecar.py + optimize sidecar format integration."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "apply_sidecar", _ROOT / "scripts" / "apply_sidecar.py"
)
apply_sidecar_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(apply_sidecar_mod)

FIXTURE_SIDECAR = _ROOT / "tests" / "fixtures" / "coding_optimized_sample.txt"
CODING_TARGET = _ROOT / "arcs" / "pipelines" / "specialists" / "coding.py"


def test_read_sidecar_strips_dspy_header():
    text = apply_sidecar_mod._read_sidecar(FIXTURE_SIDECAR)
    assert text.startswith("You are an expert coding specialist optimized")
    assert not text.startswith("#")


def test_apply_sidecar_dry_run_shows_unified_diff(capsys):
    apply_sidecar_mod.apply_sidecar(
        prompt_path=FIXTURE_SIDECAR,
        target_path=CODING_TARGET,
        dry_run=True,
    )
    out = capsys.readouterr()
    assert "--- dry-run:" in out.err
    combined = out.out + out.err
    assert "+++ " in combined or "@@" in combined
    assert "SYSTEM_PROMPT" in combined


def test_save_sidecar_format_readable_by_apply_sidecar(tmp_path):
    from arcs.optimization.dspy_common import save_sidecar_prompt

    sidecar = tmp_path / "coding_optimized.txt"
    save_sidecar_prompt(
        "Optimized prompt body for integration test.",
        sidecar,
        component_name="CODING specialist",
        source_file="arcs/pipelines/specialists/coding.py",
    )
    body = apply_sidecar_mod._read_sidecar(sidecar)
    assert body == "Optimized prompt body for integration test."


def test_cli_dry_run_exits_zero():
    proc = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "apply_sidecar.py"),
            "--prompt",
            str(FIXTURE_SIDECAR),
            "--target",
            str(CODING_TARGET),
            "--dry-run",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "@@" in proc.stdout or "SYSTEM_PROMPT" in proc.stdout
