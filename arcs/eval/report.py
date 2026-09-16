"""Render an auditable Markdown report from a saved ARCS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arcs.eval.experiments import load_experiment, sha256_file
from arcs.eval.metrics import VALID_DOMAINS, wilson_interval


def _pct(value: Any) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _ci_text(ci: dict[str, Any]) -> str:
    if ci.get("low") is None or ci.get("high") is None:
        return "n/a"
    return f"{_pct(ci['low'])}–{_pct(ci['high'])}"


def render_markdown(experiment: dict[str, Any], *, source_sha256: str | None = None) -> str:
    """Render metrics, provenance, and limitations without changing results."""
    name = str(experiment.get("name") or "ARCS experiment")
    meta = experiment.get("meta") if isinstance(experiment.get("meta"), dict) else {}
    pipeline = (
        experiment.get("pipeline")
        if isinstance(experiment.get("pipeline"), dict)
        else {}
    )
    counts = (
        pipeline.get("status_counts")
        if isinstance(pipeline.get("status_counts"), dict)
        else {}
    )
    passed = int(counts.get("PASS", 0))
    failed = int(counts.get("FAIL", 0))
    errors = int(counts.get("ERROR", 0))
    completed = passed + failed
    ci = wilson_interval(passed, completed)

    lines = [
        f"# Experiment report: {name}",
        "",
        "> Generated from the machine-readable experiment artifact. This report does not",
        "> replace human review of individual rows.",
        "",
        "## Outcome",
        "",
        "| Completed | PASS | FAIL | ERROR | PASS rate | 95% Wilson CI |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {completed} | {passed} | {failed} | {errors} | "
        f"{_pct(passed / completed if completed else None)} | {_ci_text(ci)} |",
        "",
    ]

    per_domain = pipeline.get("per_domain") if isinstance(pipeline.get("per_domain"), dict) else {}
    if per_domain:
        lines.extend(
            [
                "## Per-domain results",
                "",
                "| Domain | n | PASS rate |",
                "|---|---:|---:|",
            ]
        )
        for domain in VALID_DOMAINS:
            bucket = per_domain.get(domain) if isinstance(per_domain.get(domain), dict) else {}
            lines.append(f"| {domain} | {bucket.get('n', 0)} | {_pct(bucket.get('pass_rate'))} |")
        lines.append("")

    latency = pipeline.get("latency_ms") if isinstance(pipeline.get("latency_ms"), dict) else {}
    total_latency = latency.get("total_ms") if isinstance(latency.get("total_ms"), dict) else {}
    usage = pipeline.get("usage") if isinstance(pipeline.get("usage"), dict) else {}
    usage_totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Run ID: `{meta.get('run_id', 'unknown')}`",
            f"- Created: `{meta.get('created_at', 'unknown')}`",
            f"- Git commit: `{meta.get('git_commit', 'unknown')}`",
            f"- Dirty working tree: `{meta.get('git_dirty', 'unknown')}`",
            f"- Schema: `{meta.get('schema_version', 'legacy')}`",
            f"- Python: `{meta.get('python', 'unknown')}`",
            f"- Source artifact SHA-256: `{source_sha256 or 'not supplied'}`",
            f"- Mean / p95 latency: `{total_latency.get('mean', 'n/a')} / {total_latency.get('p95', 'n/a')} ms`",
            f"- Measured API calls / tokens: `{usage_totals.get('api_calls', 'n/a')} / {usage_totals.get('total_tokens', 'n/a')}`",
            "",
        ]
    )

    datasets = meta.get("datasets") if isinstance(meta.get("datasets"), list) else []
    if datasets:
        lines.extend(["### Dataset fingerprints", "", "| Path | Bytes | SHA-256 |", "|---|---:|---|"])
        for item in datasets:
            if isinstance(item, dict):
                lines.append(
                    f"| `{item.get('path', '?')}` | {item.get('bytes', '?')} | "
                    f"`{item.get('sha256', '?')}` |"
                )
        lines.append("")

    configuration = meta.get("configuration") if isinstance(meta.get("configuration"), dict) else {}
    if configuration:
        lines.extend(["### Non-secret configuration", "", "```json", json.dumps(configuration, indent=2, sort_keys=True), "```", ""])

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- LLM-judge PASS is a proxy metric, not a substitute for blinded human evaluation.",
            "- Report repeated runs when model sampling or provider behavior can vary.",
            "- Do not tune prompts or thresholds against a sealed test split.",
            "- Inspect and publish redacted row-level evidence before making comparative claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Run directory or experiment.json")
    parser.add_argument("--output", type=Path, help="Markdown output path")
    args = parser.parse_args(argv)

    source = args.experiment / "experiment.json" if args.experiment.is_dir() else args.experiment
    experiment = load_experiment(args.experiment)
    output = args.output or source.with_name("report.md")
    output.write_text(
        render_markdown(experiment, source_sha256=sha256_file(source)),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
