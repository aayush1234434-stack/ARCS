#!/usr/bin/env python3
"""Publish a redacted, checksummed experiment bundle under ``results/``."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arcs import config
from arcs.eval.experiments import load_experiment, sha256_file
from arcs.eval.report import render_markdown

SENSITIVE_KEY = re.compile(
    r"(^|_)(api_?key|authorization|password|credential|secret)(_|$)",
    re.IGNORECASE,
)


def redact(value: Any) -> Any:
    """Recursively remove values stored under credential-like keys."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _row_count(experiment: dict[str, Any]) -> int:
    for key in ("rows", "results", "items"):
        value = experiment.get(key)
        if isinstance(value, list):
            return len(value)
    meta = experiment.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("rows"), list):
        return len(meta["rows"])
    return 0


def publish(source: Path, destination_root: Path) -> Path:
    source_json = source / "experiment.json" if source.is_dir() else source
    experiment = redact(load_experiment(source))
    meta = experiment.get("meta") if isinstance(experiment.get("meta"), dict) else {}
    run_id = str(meta.get("run_id") or source_json.parent.name or source_json.stem)
    destination = destination_root / run_id
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; publish with a new run id"
        )
    destination.mkdir(parents=True)

    published_json = destination / "experiment.json"
    published_json.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = destination / "report.md"
    report_path.write_text(
        render_markdown(experiment, source_sha256=sha256_file(source_json)),
        encoding="utf-8",
    )

    row_count = _row_count(experiment)
    manifest = {
        "schema_version": "1.0",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_sha256": sha256_file(source_json),
        "experiment_sha256": sha256_file(published_json),
        "report_sha256": sha256_file(report_path),
        "row_level_evidence": row_count > 0,
        "row_count": row_count,
        "evidence_note": (
            "Contains redacted row-level evidence."
            if row_count
            else "Summary only; comparative claims require row-level evidence."
        ),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Run directory or experiment.json")
    parser.add_argument(
        "--destination",
        type=Path,
        default=config.PROJECT_ROOT / "results",
        help="Root directory for publishable bundles",
    )
    args = parser.parse_args(argv)
    print(publish(args.experiment, args.destination))


if __name__ == "__main__":
    main()
