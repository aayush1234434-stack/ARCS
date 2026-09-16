"""
Experiment artifact layout under ``artifacts/experiments/``.

Side effects are limited to ``save_experiment`` (and optional git metadata).
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arcs import config

EXPERIMENT_SCHEMA_VERSION = "1.0"


def _slug(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "experiment"


def make_run_id(name: str) -> str:
    """Return a filesystem-safe run id like ``2026-07-10T00-00-00_baseline-v1``."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{stamp}_{_slug(name)}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(config.PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return bool((result.stdout or "").strip())


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact or dataset."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_manifest() -> list[dict[str, Any]]:
    """Fingerprint the canonical evaluation inputs when present."""
    candidates = [
        config.DATA_DIR / "eval_queries.jsonl",
        config.ROUTER_DATA_DIR / "router_train.csv",
        config.ROUTER_DATA_DIR / "router_test.csv",
    ]
    sealed_dir = config.DATA_DIR / "sealed"
    if sealed_dir.is_dir():
        candidates.extend(sorted(sealed_dir.glob("*.jsonl")))
    manifest: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        manifest.append(
            {
                "path": str(path.relative_to(config.PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def _reproducibility_metadata() -> dict[str, Any]:
    """Capture non-secret context required to interpret a run."""
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": list(sys.argv),
        "git_dirty": _git_dirty(),
        "datasets": _data_manifest(),
        "configuration": {
            "generator_model": config.DEFAULT_GENERATOR_MODEL,
            "spec_model": config.DEFAULT_SPEC_MODEL,
            "test_generator_model": config.DEFAULT_TEST_GENERATOR_MODEL,
            "judge_model": config.DEFAULT_JUDGE_MODEL,
            "router_backend": config.ROUTER_BACKEND,
            "judge_strict": os.getenv("JUDGE_STRICT", "1"),
            "router_confidence_threshold": config.ROUTER_CONFIDENCE_THRESHOLD,
            "coding_max_retries": config.CODING_MAX_RETRIES,
        },
    }


def _format_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Experiment: {result.get('name', '?')}")
    meta = result.get("meta") or {}
    if isinstance(meta, dict):
        if meta.get("run_id"):
            lines.append(f"run_id: {meta['run_id']}")
        if meta.get("git_commit"):
            lines.append(f"git_commit: {meta['git_commit']}")
        if meta.get("created_at"):
            lines.append(f"created_at: {meta['created_at']}")
    lines.append("")

    router = result.get("router")
    if isinstance(router, dict):
        lines.append("Router")
        lines.append("-" * 40)
        acc = router.get("accuracy")
        n = router.get("n")
        lines.append(f"  n={n}  accuracy={acc if acc is None else f'{acc:.3f}'}")
        per = router.get("per_domain_accuracy") or {}
        if isinstance(per, dict):
            for domain in ("CODING", "MEDICAL", "LEGAL", "GENERAL"):
                value = per.get(domain)
                rendered = "n/a" if value is None else f"{value:.3f}"
                lines.append(f"  {domain:8s} {rendered}")
        lines.append("")

    pipeline = result.get("pipeline")
    if isinstance(pipeline, dict):
        lines.append("Pipeline")
        lines.append("-" * 40)
        lines.append(f"  n={pipeline.get('n')}")
        rates = pipeline.get("status_rates") or {}
        if isinstance(rates, dict):
            for key in ("PASS", "FAIL", "UNKNOWN", "ERROR"):
                value = rates.get(key)
                rendered = "n/a" if value is None else f"{value:.3f}"
                lines.append(f"  {key:8s} {rendered}")
        latency = pipeline.get("latency_ms") or {}
        total = latency.get("total_ms") if isinstance(latency, dict) else None
        if isinstance(total, dict) and total.get("count"):
            lines.append(
                "  latency total_ms: "
                f"mean={total.get('mean')} p50={total.get('p50')} p95={total.get('p95')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_experiment(
    result: dict[str, Any],
    *,
    name: str,
    output_dir: Path | None = None,
) -> Path:
    """Write ``experiment.json`` and ``summary.txt`` under a new run directory."""
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")

    root = Path(output_dir) if output_dir is not None else config.EXPERIMENTS_DIR
    run_id = make_run_id(name)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(result)
    meta = dict(payload.get("meta") or {})
    meta.setdefault("name", name)
    meta.setdefault("run_id", run_id)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    for key, value in _reproducibility_metadata().items():
        meta.setdefault(key, value)
    commit = _git_commit()
    if commit and "git_commit" not in meta:
        meta["git_commit"] = commit
    payload["name"] = payload.get("name") or name
    payload["meta"] = meta

    json_path = run_dir / "experiment.json"
    summary_path = run_dir / "summary.txt"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_format_summary(payload), encoding="utf-8")
    return run_dir


def load_experiment(path: Path) -> dict[str, Any]:
    """Load an experiment dict from a run dir or ``experiment.json`` path."""
    path = Path(path)
    if path.is_dir():
        path = path / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"experiment not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"experiment.json must contain an object, got {type(data).__name__}")
    return data


def latest_experiment(output_dir: Path | None = None) -> Path | None:
    """Return the newest experiment run directory, or None if none exist."""
    root = Path(output_dir) if output_dir is not None else config.EXPERIMENTS_DIR
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "experiment.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)
