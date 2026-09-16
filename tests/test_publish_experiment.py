from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_experiment import publish, redact


def test_redact_preserves_token_metrics_but_removes_credentials():
    result = redact(
        {
            "api_key": "secret",
            "total_tokens": 42,
            "nested": {"authorization": "Bearer secret"},
        }
    )
    assert result["api_key"] == "[REDACTED]"
    assert result["total_tokens"] == 42
    assert result["nested"]["authorization"] == "[REDACTED]"


def test_publish_creates_checksummed_bundle(tmp_path: Path):
    source = tmp_path / "run" / "experiment.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "name": "sample",
                "meta": {"run_id": "sample-run"},
                "pipeline": {"status_counts": {"PASS": 1, "FAIL": 0, "ERROR": 0}},
                "rows": [{"query": "hello", "status": "PASS"}],
            }
        ),
        encoding="utf-8",
    )

    destination = publish(source, tmp_path / "published")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_level_evidence"] is True
    assert manifest["row_count"] == 1
    assert len(manifest["experiment_sha256"]) == 64
    assert (destination / "report.md").exists()

    with pytest.raises(FileExistsError):
        publish(source, tmp_path / "published")
