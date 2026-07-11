"""snapshot_post_fix default input discovery."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "snapshot_post_fix", _ROOT / "scripts" / "snapshot_post_fix.py"
)
snapshot_post_fix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(snapshot_post_fix)


def _write_experiment_dir(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(
        json.dumps({"pipeline": {"rows": []}}),
        encoding="utf-8",
    )
    return directory


def test_matches_domain_dir_excludes_merged_and_resume():
    assert snapshot_post_fix._matches_domain_dir(
        "2026-07-11T09-35-04_post-fix-coding-v2", "coding"
    )
    assert not snapshot_post_fix._matches_domain_dir(
        "2026-07-11T10-29-30_post-fix-v2-merged", "coding"
    )
    assert not snapshot_post_fix._matches_domain_dir(
        "2026-07-11T10-29-29_post-fix-resume-v1", "general"
    )


def test_resolve_default_inputs_finds_domain_runs_and_resume(monkeypatch, tmp_path):
    """CI-safe: synthetic experiment dirs; no dependency on local artifacts/."""
    exp_root = tmp_path / "experiments"
    for tag in snapshot_post_fix.DOMAIN_TAGS:
        _write_experiment_dir(exp_root, f"2026-07-11T09-00-00_post-fix-{tag}-v1")
    _write_experiment_dir(exp_root, "2026-07-11T10-00-00_post-fix-resume-v1")

    monkeypatch.setattr(snapshot_post_fix.config, "EXPERIMENTS_DIR", exp_root)

    domain_paths, missing, resume = snapshot_post_fix._resolve_default_inputs()
    assert not missing
    names = " ".join(p.name.lower() for p in domain_paths)
    for tag in snapshot_post_fix.DOMAIN_TAGS:
        assert tag in names
    assert resume is not None
    assert "resume" in resume.name.lower()
    assert len(domain_paths) >= len(snapshot_post_fix.DOMAIN_TAGS)


@pytest.mark.skipif(
    not snapshot_post_fix.config.EXPERIMENTS_DIR.exists(),
    reason="local artifacts/experiments not present",
)
def test_resolve_default_inputs_local_artifacts_when_present():
    """Optional integration check when developer has saved eval runs locally."""
    domain_paths, missing, resume = snapshot_post_fix._resolve_default_inputs()
    if missing:
        pytest.skip(f"missing local post-fix runs: {missing}")
    names = " ".join(p.name.lower() for p in domain_paths)
    for tag in snapshot_post_fix.DOMAIN_TAGS:
        assert tag in names
    assert resume is not None
