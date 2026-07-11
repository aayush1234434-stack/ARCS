"""Dry-run tests for scripts/eval_judge_modes.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/experiments/2026-07-11T12-36-52_post-fix-v2-merged"


def test_eval_judge_modes_compare_on_cached_experiment():
    if not (ARTIFACT / "experiment.json").is_file():
        return
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval_judge_modes.py"),
            "--compare",
            str(ARTIFACT),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FAIL→PASS flips (relaxed only): 0" in proc.stdout
    assert "MEDICAL" in proc.stdout
