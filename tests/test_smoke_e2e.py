"""Smoke e2e script — dry-run and case loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.smoke_e2e import (
    EXPECTED_DOMAINS,
    SMOKE_IDS,
    load_smoke_cases,
    run_smoke,
)


def test_smoke_ids_cover_four_domains_plus_coding_prose():
    domains = [EXPECTED_DOMAINS[i] for i in SMOKE_IDS]
    assert domains.count("LEGAL") == 1
    assert domains.count("MEDICAL") == 1
    assert domains.count("GENERAL") == 1
    assert domains.count("CODING") == 2
    assert "eval-042" in SMOKE_IDS


def test_load_smoke_cases_from_eval_file():
    cases = load_smoke_cases()
    assert len(cases) == 5
    assert cases[0].id == "eval-024"
    assert cases[0].expected_domain == "LEGAL"
    assert "will" in cases[0].query.lower() or "trust" in cases[0].query.lower()


def test_run_smoke_dry_run():
    report = run_smoke(dry_run=True)
    assert report["dry_run"] is True
    assert len(report["cases"]) == 5
    assert report["summary"]["planned"] == 5


def test_main_dry_run_json(capsys, monkeypatch):
    from scripts import smoke_e2e as mod

    monkeypatch.setattr(sys, "argv", ["smoke_e2e.py", "--dry-run", "--json"])
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["cases"]) == 5
