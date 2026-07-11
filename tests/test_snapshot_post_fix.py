"""snapshot_post_fix default input discovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "snapshot_post_fix", _ROOT / "scripts" / "snapshot_post_fix.py"
)
snapshot_post_fix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(snapshot_post_fix)


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


def test_resolve_default_inputs_finds_domain_runs_and_resume():
    domain_paths, missing, resume = snapshot_post_fix._resolve_default_inputs()
    assert not missing
    names = " ".join(p.name.lower() for p in domain_paths)
    for tag in snapshot_post_fix.DOMAIN_TAGS:
        assert tag in names
    assert resume is not None
    assert "resume" in resume.name.lower()
    assert len(domain_paths) >= len(snapshot_post_fix.DOMAIN_TAGS)
