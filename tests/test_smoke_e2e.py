"""Smoke e2e script — dry-run and case loading (no live API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.main import PipelineError
from scripts.smoke_e2e import (
    EXIT_OK,
    EXIT_SETUP,
    EXIT_SMOKE_FAIL,
    EXPECTED_DOMAINS,
    SMOKE_IDS,
    _exit_code_for_report,
    _run_case,
    _summarize_results,
    load_smoke_cases,
    run_smoke,
)


def test_smoke_ids_cover_four_domains_plus_coding_prose():
    domains = [EXPECTED_DOMAINS[i] for i in SMOKE_IDS]
    assert len(SMOKE_IDS) == 5
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
    assert report["summary"]["errors"] == 0


def test_summarize_results_requires_zero_errors_for_success():
    ok = _summarize_results(
        [
            {"status": "PASS"},
            {"status": "FAIL"},
            {"status": "PASS"},
            {"status": "FAIL"},
            {"status": "PASS"},
        ],
        total=5,
    )
    assert ok["errors"] == 0
    assert ok["success"] is True
    assert ok["pass"] == 3
    assert ok["fail"] == 2

    bad = _summarize_results(
        [
            {"status": "PASS"},
            {"status": "ERROR", "error_class": "rate_limit"},
            {"status": "FAIL"},
            {"status": "FAIL"},
            {"status": "UNKNOWN", "error_class": "unknown"},
        ],
        total=5,
    )
    assert bad["errors"] == 2
    assert bad["success"] is False
    assert bad["error_classes"] == {"rate_limit": 1, "unknown": 1}


def test_run_case_pipeline_error_includes_error_class(monkeypatch):
    from scripts import smoke_e2e as mod

    case = load_smoke_cases()[0]

    def _fail(_query: str):
        raise PipelineError(
            {
                "error": "Rate limit reached for tokens per day (TPD)",
                "error_class": "rate_limit",
            }
        )

    monkeypatch.setattr(mod, "run_pipeline", _fail)
    row = _run_case(case)
    assert row["status"] == "ERROR"
    assert row["error_class"] == "rate_limit"


def test_exit_codes():
    assert (
        _exit_code_for_report(
            {"summary": {"errors": 0, "complete": 5, "total": 5}},
            dry_run=False,
        )
        == EXIT_OK
    )
    assert (
        _exit_code_for_report(
            {"summary": {"errors": 1, "complete": 4, "total": 5}},
            dry_run=False,
        )
        == EXIT_SMOKE_FAIL
    )
    assert _exit_code_for_report({"summary": {}}, dry_run=True) == EXIT_OK


def test_main_dry_run_json(capsys, monkeypatch):
    from scripts import smoke_e2e as mod

    monkeypatch.setattr(sys, "argv", ["smoke_e2e.py", "--dry-run", "--json"])
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["cases"]) == 5
    assert payload["exit_code"] == EXIT_OK


def test_main_setup_failure_exit_code(monkeypatch):
    from scripts import smoke_e2e as mod

    monkeypatch.setattr(sys, "argv", ["smoke_e2e.py", "--json"])
    monkeypatch.setattr(mod, "run_smoke", lambda **_: (_ for _ in ()).throw(RuntimeError("no keys")))
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == EXIT_SETUP
