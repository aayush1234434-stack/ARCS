"""
Experiment artifact layout under ``artifacts/experiments/``.

Side effects are limited to ``save_experiment`` (and optional git metadata).
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arcs import config


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
