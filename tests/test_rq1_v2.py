"""Tests for RQ1 v2 (real-feedback corpus) helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_rq1_corpus import build_real_corpus
from scripts.rq1_run import (
    _paired_misroute_comparison,
    _require_real_corpus_ready,
    _resolve_corpus,
)


def test_resolve_corpus_real_and_bootstrap() -> None:
    path, kind = _resolve_corpus("real")
    assert kind == "real"
    assert path.name == "feedback_corpus_real.jsonl"

    path, kind = _resolve_corpus("bootstrap")
    assert kind == "bootstrap"
    assert path.name == "feedback_corpus.jsonl"


def test_build_real_corpus_from_requests(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    rows = [
        {
            "query_id": "demo-1",
            "query": "What is fair use?",
            "user_feedback": "NEGATIVE",
            "correct_domain": "LEGAL",
            "route": {"domain": "MEDICAL", "confidence": 0.45},
            "verification": {"verdict": "FAIL", "score": 0.2},
        },
        {
            "query_id": "demo-2",
            "query": "Write fibonacci in Python",
            "user_feedback": "NEGATIVE",
            "correct_domain": "CODING",
            "route": {"domain": "CODING", "confidence": 0.9},
            "verification": {"verdict": "FAIL", "score": 0.1},
        },
    ]
    with requests.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    corpus = build_real_corpus(requests_path=requests)
    assert len(corpus) == 2
    components = {r["attribution"]["component"] for r in corpus}
    assert "ROUTER" in components
    assert all(r["user_feedback"] == "NEGATIVE" for r in corpus)
    assert all(r["source"] == "demo_feedback" for r in corpus)


def test_require_real_corpus_ready_exits_when_too_small(tmp_path: Path) -> None:
    corpus = tmp_path / "feedback_corpus_real.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "query": "q",
                    "correct_domain": "LEGAL",
                    "attribution": {"component": "ROUTER"},
                }
            )
            + "\n"
        )

    with pytest.raises(SystemExit) as exc:
        _require_real_corpus_ready(corpus)
    assert exc.value.code == 1


def test_paired_misroute_comparison_counts() -> None:
    pre = {
        "eval_queries_router_accuracy": {
            "rows": [
                {"id": "eval-001", "expected_domain": "LEGAL", "predicted_domain": "MEDICAL"},
                {"id": "eval-002", "expected_domain": "CODING", "predicted_domain": "CODING"},
            ]
        }
    }
    run_a = {
        "eval_queries_router_accuracy": {
            "rows": [
                {"id": "eval-001", "expected_domain": "LEGAL", "predicted_domain": "LEGAL"},
                {"id": "eval-002", "expected_domain": "CODING", "predicted_domain": "CODING"},
            ]
        }
    }
    run_b = {
        "eval_queries_router_accuracy": {
            "rows": [
                {"id": "eval-001", "expected_domain": "LEGAL", "predicted_domain": "MEDICAL"},
                {"id": "eval-002", "expected_domain": "CODING", "predicted_domain": "CODING"},
            ]
        }
    }

    result = _paired_misroute_comparison(pre, run_a, run_b)
    assert result["pre_misroute_count"] == 1
    assert result["misroutes_fixed_by_run_a_vs_pre"] == 1
    assert result["misroutes_fixed_by_run_b_vs_pre"] == 0
    assert result["mcnemar_discordant_a_only"] == 1
    assert result["mcnemar_discordant_b_only"] == 0
    assert "McNemar-style" in result["note"]
