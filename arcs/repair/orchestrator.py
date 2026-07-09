"""
Repair orchestrator — one entry point over existing queue / export / optimize tools.

Default flow:
  1. extract_queues (unless skipped)
  2. ROUTER → export_router_examples (+ optional DistilBERT train)
  3. SPECIALIST → suggest (or run) domain DSPy optimize scripts
  4. VERIFIER → suggest (or run) optimize_judge
  5. AMBIGUOUS → summary only (never train)

DSPy sidecars are never auto-applied to source modules.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from arcs import config
from arcs.post.queues import (
    COMPONENTS as QUEUE_COMPONENTS,
    extract_queues,
    format_summary,
    queue_counts,
)
from arcs.router.export_training import (
    DEFAULT_CSV as ROUTER_CSV,
    DEFAULT_QUEUE as ROUTER_QUEUE,
    export_router_examples,
)

COMPONENTS = QUEUE_COMPONENTS  # ("ROUTER", "SPECIALIST", "VERIFIER", "AMBIGUOUS")

_QUEUE_FILENAMES = {
    "ROUTER": "router_queue.jsonl",
    "SPECIALIST": "specialist_queue.jsonl",
    "VERIFIER": "verifier_queue.jsonl",
    "AMBIGUOUS": "ambiguous_queue.jsonl",
}

_SPECIALIST_OPTIMIZE_SCRIPTS: dict[str, str] = {
    "MEDICAL": "scripts/optimize_medical.py",
    "CODING": "scripts/optimize_coding.py",
    "LEGAL": "scripts/optimize_legal.py",
    "GENERAL": "scripts/optimize_general.py",
}

_SUPPORTED_DSPY_DOMAINS = frozenset(_SPECIALIST_OPTIMIZE_SCRIPTS)


def _queues_dir() -> Path:
    return config.LOGS_DIR / "queues"


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _load_queue_counts_from_disk() -> dict[str, int]:
    root = _queues_dir()
    return {
        component: _count_jsonl(root / _QUEUE_FILENAMES[component])
        for component in COMPONENTS
    }


def _selected(components: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not components:
        return COMPONENTS
    selected: list[str] = []
    for raw in components:
        name = str(raw).strip().upper()
        if name not in COMPONENTS:
            raise ValueError(
                f"Unknown component {raw!r}. Choose from: {', '.join(COMPONENTS)}"
            )
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _suggest(cmd: str) -> dict[str, str]:
    return {"action": "suggest", "command": cmd}


def _run_module(module: str, *args: str) -> dict[str, Any]:
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "action": "run",
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "ok": result.returncode == 0,
    }


def _run_script(script_rel: str, *args: str) -> dict[str, Any]:
    script = config.PROJECT_ROOT / script_rel
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "action": "run",
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "ok": result.returncode == 0,
    }


def _repair_router(
    *,
    dry_run: bool,
    train_router: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queue_count": counts.get("ROUTER", 0),
        "rows_exported": 0,
        "suggested": [],
        "ran": [],
        "errors": [],
        "notes": [],
    }

    if result["queue_count"] == 0:
        result["notes"].append("ROUTER queue empty — nothing to export.")
        return result

    result["suggested"].append(
        "python scripts/label_router_failures.py --list"
    )
    result["suggested"].append(
        "python scripts/label_router_failures.py --interactive"
    )

    if dry_run:
        result["notes"].append(
            f"Dry-run: would export labeled rows from {ROUTER_QUEUE} → {ROUTER_CSV}"
        )
        result["suggested"].append("python scripts/retrain_router.py --skip-extract")
        if train_router:
            result["notes"].append(
                "Dry-run: would also run python -m arcs.router.train"
            )
        return result

    try:
        written = export_router_examples(
            queue_path=ROUTER_QUEUE,
            output_csv=ROUTER_CSV,
            append=True,
        )
    except FileNotFoundError as exc:
        result["errors"].append(str(exc))
        result["suggested"].append("python scripts/extract_queues.py")
        return result
    except Exception as exc:  # noqa: BLE001 — surface in summary
        result["errors"].append(f"export_router_examples failed: {exc}")
        return result

    result["rows_exported"] = written
    result["ran"].append(
        {
            "action": "export_router_examples",
            "queue": str(ROUTER_QUEUE),
            "csv": str(ROUTER_CSV),
            "rows_exported": written,
        }
    )

    if written == 0:
        result["notes"].append(
            "No new labeled ROUTER rows exported. "
            "Label failures first with scripts/label_router_failures.py"
        )
        result["suggested"].append("python scripts/retrain_router.py --skip-extract")
        return result

    if train_router:
        train_result = _run_module("arcs.router.train")
        result["ran"].append(train_result)
        if not train_result["ok"]:
            result["errors"].append(
                f"router train failed (exit {train_result['returncode']})"
            )
    else:
        result["suggested"].append(
            "python scripts/retrain_router.py --skip-extract"
        )
        result["notes"].append(
            "Exported training rows; pass --train-router to retrain DistilBERT."
        )

    return result


def _specialist_domains_in_queue() -> list[str]:
    path = _queues_dir() / _QUEUE_FILENAMES["SPECIALIST"]
    if not path.exists():
        return []
    domains: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            pipeline = record.get("pipeline")
            if isinstance(pipeline, dict) and pipeline.get("pipeline_id"):
                domains.add(str(pipeline["pipeline_id"]).upper())
                continue
            specialist = record.get("specialist")
            if isinstance(specialist, dict):
                if specialist.get("pipeline_id"):
                    domains.add(str(specialist["pipeline_id"]).upper())
                elif specialist.get("domain"):
                    domains.add(str(specialist["domain"]).upper())
    return sorted(domains)


def _repair_specialist(
    *,
    dry_run: bool,
    run_dspy: bool,
    domain: str | None,
    counts: dict[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queue_count": counts.get("SPECIALIST", 0),
        "domains_seen": _specialist_domains_in_queue(),
        "suggested": [],
        "ran": [],
        "errors": [],
        "notes": [],
    }

    if result["queue_count"] == 0:
        result["notes"].append("SPECIALIST queue empty — nothing to optimize.")
        return result

    target_domains: list[str]
    if domain:
        target = domain.strip().upper()
        if target not in _SUPPORTED_DSPY_DOMAINS:
            result["errors"].append(
                f"No DSPy optimize script for domain {target!r}. "
                f"Supported today: {', '.join(sorted(_SUPPORTED_DSPY_DOMAINS))}"
            )
            for name, script in _SPECIALIST_OPTIMIZE_SCRIPTS.items():
                result["suggested"].append(f"python {script} --dry-run")
            return result
        target_domains = [target]
    else:
        # Prefer domains present in the queue that we know how to optimize;
        # always surface MEDICAL instructions as the documented path.
        seen = [d for d in result["domains_seen"] if d in _SUPPORTED_DSPY_DOMAINS]
        target_domains = seen or ["MEDICAL"]
        unknown = [d for d in result["domains_seen"] if d not in _SUPPORTED_DSPY_DOMAINS]
        if unknown:
            result["notes"].append(
                "SPECIALIST domains without a DSPy script yet: "
                + ", ".join(unknown)
            )

    for target in target_domains:
        script = _SPECIALIST_OPTIMIZE_SCRIPTS[target]
        if dry_run or not run_dspy:
            result["suggested"].append(f"python {script} --dry-run")
            result["suggested"].append(f"python {script}")
            result["notes"].append(
                f"{target}: DSPy writes a sidecar under artifacts/prompts/ — "
                "review and apply SYSTEM_PROMPT manually (never auto-applied)."
            )
            if not run_dspy and not dry_run:
                result["notes"].append(
                    f"Pass --run-dspy --domain {target} to execute {script}."
                )
            continue

        run_result = _run_script(script)
        result["ran"].append(run_result)
        if not run_result["ok"]:
            result["errors"].append(
                f"{script} failed (exit {run_result['returncode']})"
            )
        else:
            result["notes"].append(
                f"{target}: DSPy sidecar written for human review — "
                "do not auto-apply to source."
            )

    return result


def _repair_verifier(
    *,
    dry_run: bool,
    run_dspy: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queue_count": counts.get("VERIFIER", 0),
        "suggested": [],
        "ran": [],
        "errors": [],
        "notes": [],
    }
    script = "scripts/optimize_judge.py"

    if result["queue_count"] == 0:
        result["notes"].append("VERIFIER queue empty — nothing to optimize.")
        return result

    if dry_run or not run_dspy:
        result["suggested"].append(f"python {script} --dry-run")
        result["suggested"].append(f"python {script}")
        result["notes"].append(
            "VERIFIER: DSPy writes artifacts/prompts/judge_optimized.txt — "
            "review and apply SYSTEM_PROMPT in judge.py manually."
        )
        if not run_dspy and not dry_run:
            result["notes"].append(f"Pass --run-dspy to execute {script}.")
        return result

    run_result = _run_script(script)
    result["ran"].append(run_result)
    if not run_result["ok"]:
        result["errors"].append(
            f"{script} failed (exit {run_result['returncode']})"
        )
    else:
        result["notes"].append(
            "Judge sidecar written for human review — do not auto-apply."
        )
    return result


def _repair_ambiguous(*, counts: dict[str, int]) -> dict[str, Any]:
    count = counts.get("AMBIGUOUS", 0)
    path = _queues_dir() / _QUEUE_FILENAMES["AMBIGUOUS"]
    return {
        "queue_count": count,
        "queue_path": str(path),
        "suggested": [],
        "ran": [],
        "errors": [],
        "notes": [
            "AMBIGUOUS failures are for human review only — do not train on them.",
            f"Inspect {path}" if count else "AMBIGUOUS queue empty.",
        ],
    }


def repair_all(
    *,
    extract_first: bool = True,
    dry_run: bool = False,
    components: tuple[str, ...] | list[str] | None = None,
    train_router: bool = False,
    run_dspy: bool = False,
    domain: str | None = None,
) -> dict[str, Any]:
    """Sort failure queues and run (or suggest) the repair path per component.

    Args:
        extract_first: When True, call ``extract_queues()`` first.
        dry_run: Plan / suggest only; do not export, train, or run DSPy.
        components: Subset of COMPONENTS to process (default: all).
        train_router: When True and ROUTER is selected, run DistilBERT train
            after a successful export (ignored in dry_run).
        run_dspy: When True, execute optimize_* scripts instead of only
            printing instructions (ignored in dry_run).
        domain: Specialist domain for DSPy (e.g. MEDICAL). Used when
            SPECIALIST is selected.

    Returns:
        Summary dict with queue counts, per-component results, and errors.
    """
    selected = _selected(components)
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "extract_first": extract_first,
        "components": list(selected),
        "train_router": train_router,
        "run_dspy": run_dspy,
        "domain": domain.strip().upper() if domain else None,
        "queue_counts": {},
        "extract": None,
        "results": {},
        "errors": [],
        "scripts_suggested": [],
        "scripts_run": [],
    }

    if extract_first:
        try:
            queues = extract_queues(dry_run=dry_run)
            counts = queue_counts(queues)
            summary["extract"] = {
                "ok": True,
                "dry_run": dry_run,
                "summary": format_summary(counts),
            }
            summary["queue_counts"] = counts
        except FileNotFoundError as exc:
            summary["extract"] = {"ok": False, "error": str(exc)}
            summary["errors"].append(str(exc))
            # Fall back to on-disk queues if present so other steps can still plan.
            summary["queue_counts"] = _load_queue_counts_from_disk()
        except Exception as exc:  # noqa: BLE001
            summary["extract"] = {"ok": False, "error": str(exc)}
            summary["errors"].append(f"extract_queues failed: {exc}")
            summary["queue_counts"] = _load_queue_counts_from_disk()
    else:
        summary["extract"] = {"ok": True, "skipped": True}
        summary["queue_counts"] = _load_queue_counts_from_disk()

    counts = summary["queue_counts"]

    for component in selected:
        if component == "ROUTER":
            result = _repair_router(
                dry_run=dry_run,
                train_router=train_router,
                counts=counts,
            )
        elif component == "SPECIALIST":
            result = _repair_specialist(
                dry_run=dry_run,
                run_dspy=run_dspy,
                domain=domain,
                counts=counts,
            )
        elif component == "VERIFIER":
            result = _repair_verifier(
                dry_run=dry_run,
                run_dspy=run_dspy,
                counts=counts,
            )
        elif component == "AMBIGUOUS":
            result = _repair_ambiguous(counts=counts)
        else:
            continue

        summary["results"][component] = result
        summary["scripts_suggested"].extend(result.get("suggested") or [])
        for item in result.get("ran") or []:
            if isinstance(item, dict) and item.get("command"):
                summary["scripts_run"].append(item["command"])
            elif isinstance(item, dict) and item.get("action"):
                summary["scripts_run"].append(str(item["action"]))
        summary["errors"].extend(result.get("errors") or [])

    # Deduplicate suggested commands while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for cmd in summary["scripts_suggested"]:
        if cmd not in seen:
            seen.add(cmd)
            deduped.append(cmd)
    summary["scripts_suggested"] = deduped

    return summary
