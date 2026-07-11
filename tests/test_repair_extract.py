"""Tests for combined feedback log extraction in repair / extract_queues."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcs.post.queues import extract_queues, feedback_log_paths, queue_counts


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _negative_record(query_id: str, component: str) -> dict:
    return {
        "query_id": query_id,
        "query": f"q-{query_id}",
        "user_feedback": "NEGATIVE",
        "attribution": {"component": component},
        "pipeline": {"pipeline_id": "CODING"},
    }


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch):
    logs = tmp_path / "logs"
    queues = logs / "queues"
    monkeypatch.setattr("arcs.config.LOGS_DIR", logs)
    monkeypatch.setattr("arcs.post.queues.REQUESTS_LOG", logs / "requests.jsonl")
    monkeypatch.setattr("arcs.post.queues.EVAL_FAILURES_LOG", logs / "eval_failures.jsonl")
    monkeypatch.setattr("arcs.post.queues.config.LOGS_DIR", logs)
    return logs, queues


def test_feedback_log_paths_merges_both_when_present(log_dir):
    logs, _ = log_dir
    _write_jsonl(logs / "requests.jsonl", [_negative_record("live-1", "AMBIGUOUS")])
    _write_jsonl(logs / "eval_failures.jsonl", [_negative_record("eval-005", "SPECIALIST")])

    paths = feedback_log_paths(include_eval_failures=True)
    assert len(paths) == 2
    assert paths[0].name == "requests.jsonl"
    assert paths[1].name == "eval_failures.jsonl"


def test_extract_queues_include_eval_failures_counts_specialist(log_dir):
    logs, queues_dir = log_dir
    _write_jsonl(logs / "requests.jsonl", [_negative_record("live-1", "AMBIGUOUS")])
    _write_jsonl(
        logs / "eval_failures.jsonl",
        [
            _negative_record("eval-005", "SPECIALIST"),
            _negative_record("eval-006", "SPECIALIST"),
        ],
    )

    queues = extract_queues(dry_run=True, include_eval_failures=True)
    counts = queue_counts(queues)
    assert counts["SPECIALIST"] == 2
    assert counts["AMBIGUOUS"] == 1


def test_extract_queues_requests_only_ignores_eval_failures(log_dir):
    logs, _ = log_dir
    _write_jsonl(logs / "requests.jsonl", [_negative_record("live-1", "AMBIGUOUS")])
    _write_jsonl(logs / "eval_failures.jsonl", [_negative_record("eval-005", "SPECIALIST")])

    queues = extract_queues(dry_run=True, include_eval_failures=False)
    counts = queue_counts(queues)
    assert counts["SPECIALIST"] == 0
    assert counts["AMBIGUOUS"] == 1
