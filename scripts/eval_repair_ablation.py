"""RQ1-bis: fast router repair ablation on held-out eval queries.

Compares three router checkpoints on eval_queries router accuracy only
(no full pipeline — local DistilBERT routing is fast):

  Arm 0 — production / pre-RQ1 backup (baseline before repair retrain)
  Arm 1 — router after RQ1 Run A (all negative feedback)
  Arm 2 — router after RQ1 Run B (ROUTER-attributed feedback only)

Writes a manifest under artifacts/experiments/<ts>_repair_ablation/manifest.json
with a table: arm | router_eval_acc | delta vs arm0.

Usage:
    python scripts/eval_repair_ablation.py                 # dry-run (default)
    python scripts/eval_repair_ablation.py --execute
    python scripts/eval_repair_ablation.py --execute --limit 12 --domains LEGAL
    python scripts/eval_repair_ablation.py --execute --corpus real   # v2 model dirs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.eval.metrics import VALID_DOMAINS, router_accuracy

DEFAULT_EVAL_QUERIES = config.DATA_DIR / "eval_queries.jsonl"

LIVE_MODEL_DIR = config.ROUTER_MODEL_DIR
PRE_BACKUP_DIR = config.ARTIFACTS_DIR / "router-model-pre-rq1"
RUN_A_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-a"
RUN_B_MODEL_DIR = config.ARTIFACTS_DIR / "router-model-rq1-run-b"
RUN_A_MODEL_DIR_V2 = config.ARTIFACTS_DIR / "router-model-rq1-v2-run-a"
RUN_B_MODEL_DIR_V2 = config.ARTIFACTS_DIR / "router-model-rq1-v2-run-b"

KIND = "repair_ablation"
DOMAIN_SET = frozenset(VALID_DOMAINS)


def _load_eval_query_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(row, dict):
                continue
            query = row.get("query")
            expected = row.get("expected_domain")
            if not isinstance(query, str) or not query.strip():
                continue
            if not isinstance(expected, str) or not expected.strip():
                continue
            rows.append(row)
    return rows


def _normalize_domain(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if normalized in DOMAIN_SET else None


def _parse_csv_set(raw: str | None, *, label: str) -> set[str] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} filter is empty")
    return set(parts)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    ids: set[str] | None,
    domains: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if ids is not None:
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id not in ids:
                continue
        if domains is not None:
            if _normalize_domain(row.get("expected_domain")) not in domains:
                continue
        filtered.append(row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _resolve_arm_models(*, corpus: str) -> dict[str, dict[str, Any]]:
    """Return arm definitions with model_dir paths."""
    if corpus == "real":
        run_a_dir, run_b_dir = RUN_A_MODEL_DIR_V2, RUN_B_MODEL_DIR_V2
        suffix = " (RQ1 v2 real feedback)"
    else:
        run_a_dir, run_b_dir = RUN_A_MODEL_DIR, RUN_B_MODEL_DIR
        suffix = " (RQ1 bootstrap)"

    arm0_dir = PRE_BACKUP_DIR if PRE_BACKUP_DIR.exists() else LIVE_MODEL_DIR
    arm0_label = (
        "pre-rq1 backup"
        if PRE_BACKUP_DIR.exists()
        else "live production router"
    )

    return {
        "arm0": {
            "label": arm0_label,
            "description": "Production router before RQ1 retrain (or live model if no backup)",
            "model_dir": arm0_dir,
        },
        "arm1": {
            "label": f"Run A — all negatives{suffix}",
            "description": "Router retrained on base + all corpus negatives",
            "model_dir": run_a_dir,
        },
        "arm2": {
            "label": f"Run B — ROUTER-only{suffix}",
            "description": "Router retrained on base + ROUTER-attributed negatives only",
            "model_dir": run_b_dir,
        },
    }


def _score_subset(
    rows: list[dict[str, Any]],
    *,
    model_dir: Path,
) -> dict[str, Any]:
    """Score only the filtered eval-query rows with a given model checkpoint."""
    from arcs.router.classifier import route

    scored: list[dict[str, Any]] = []
    for row in rows:
        query = str(row["query"]).strip()
        expected = str(row["expected_domain"]).strip().upper()
        try:
            result = route(query, model_dir=str(model_dir))
            predicted = str(result.get("domain") or "").strip().upper()
            confidence = result.get("confidence")
        except Exception as exc:  # noqa: BLE001 — keep ablation going
            print(
                f"Warning: route failed for {row.get('id')!r}: {exc}",
                file=sys.stderr,
            )
            predicted = None
            confidence = None
        scored.append(
            {
                "id": row.get("id"),
                "query": query,
                "expected_domain": expected,
                "predicted_domain": predicted,
                "router_confidence": confidence,
            }
        )

    metrics = router_accuracy(scored)
    return {
        "n_queries": len(scored),
        "metrics": metrics,
        "rows": scored,
    }


def _eval_queries_acc(result: dict[str, Any]) -> float | None:
    metrics = result.get("metrics") or {}
    value = metrics.get("accuracy")
    return None if value is None else float(value)


def _fmt_acc(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) < 1e-9:
        return "=0.0000"
    return f"{value:+.4f}"


def _build_table(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    arm0_acc = _eval_queries_acc(arms["arm0"].get("eval_result") or {})
    table: list[dict[str, Any]] = []
    for arm_id in ("arm0", "arm1", "arm2"):
        arm = arms[arm_id]
        acc = _eval_queries_acc(arm.get("eval_result") or {})
        delta = None if acc is None or arm0_acc is None else acc - arm0_acc
        table.append(
            {
                "arm": arm_id,
                "label": arm["label"],
                "model_dir": str(arm["model_dir"]),
                "router_eval_acc": acc,
                "delta_vs_arm0": delta if arm_id != "arm0" else 0.0,
            }
        )
    return table


def _pick_winner(table: list[dict[str, Any]]) -> str:
    candidates = [
        (row["arm"], row["router_eval_acc"])
        for row in table
        if row["arm"] != "arm0" and row["router_eval_acc"] is not None
    ]
    if not candidates:
        return "none"
    best_acc = max(acc for _, acc in candidates)
    winners = [arm for arm, acc in candidates if abs(acc - best_acc) < 1e-9]
    if len(winners) > 1:
        return "tie"
    return winners[0]


def _print_plan(
    *,
    arms: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    eval_path: Path,
    corpus: str,
) -> None:
    print("RQ1-bis repair ablation plan (dry-run — re-run with --execute)\n")
    print(f"Eval queries: {eval_path}  ({len(rows)} row(s) after filters)")
    print(f"Corpus kind:  {corpus}")
    print("\nArms (eval_queries router accuracy only — no full pipeline):")
    for arm_id, arm in arms.items():
        path = arm["model_dir"]
        mark = "OK" if path.exists() else "MISSING"
        print(f"  {arm_id}: {arm['label']}")
        print(f"       [{mark}] {path}")
    print("\nOn --execute, writes:")
    print("  artifacts/experiments/<timestamp>_repair_ablation/manifest.json")
    print("\nTable columns: arm | router_eval_acc | delta_vs_arm0")


def _print_table(table: list[dict[str, Any]]) -> None:
    print("\n=== RQ1-bis router ablation (eval_queries accuracy) ===", file=sys.stderr)
    header = f"{'arm':6s} {'router_eval_acc':>16s} {'delta_vs_arm0':>14s}  label"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for row in table:
        print(
            f"{row['arm']:6s} {_fmt_acc(row['router_eval_acc']):>16s} "
            f"{_fmt_delta(row['delta_vs_arm0']):>14s}  {row['label']}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1-bis: compare pre-RQ1 vs Run A vs Run B router checkpoints "
            "on eval_queries routing accuracy (fast, no full pipeline)."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_EVAL_QUERIES,
        help=f"Eval queries JSONL (default: {DEFAULT_EVAL_QUERIES})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate at most N rows after filters",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated eval ids to include",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated expected domains (e.g. CODING,LEGAL)",
    )
    parser.add_argument(
        "--corpus",
        choices=("bootstrap", "real"),
        default="bootstrap",
        help="Which RQ1 retrain checkpoints to use (default: bootstrap)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run routing eval for all arms and write manifest (default: dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only (default unless --execute)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print manifest JSON to stdout after execute",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        print("Error: --limit must be >= 0", file=sys.stderr)
        sys.exit(1)

    try:
        id_filter = _parse_csv_set(args.ids, label="--ids")
        domain_filter = _parse_csv_set(args.domains, label="--domains")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if domain_filter is not None:
        normalized: set[str] = set()
        for raw in domain_filter:
            domain = _normalize_domain(raw)
            if domain is None:
                print(
                    f"Error: invalid domain in --domains: {raw!r} "
                    f"(choose from {', '.join(VALID_DOMAINS)})",
                    file=sys.stderr,
                )
                sys.exit(1)
            normalized.add(domain)
        domain_filter = normalized

    if not args.input.exists():
        print(f"Error: eval file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = _load_eval_query_rows(args.input)
    rows = _filter_rows(rows, ids=id_filter, domains=domain_filter, limit=args.limit)
    if not rows:
        print("Error: no eval rows matched filters", file=sys.stderr)
        sys.exit(1)

    arms = _resolve_arm_models(corpus=args.corpus)

    # --dry-run explicit, or default unless --execute
    if args.dry_run or not args.execute:
        _print_plan(arms=arms, rows=rows, eval_path=args.input, corpus=args.corpus)
        return

    missing = [
        f"{arm_id}: {arm['model_dir']}"
        for arm_id, arm in arms.items()
        if not arm["model_dir"].exists()
    ]
    if missing:
        print("Error: missing model checkpoint(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        if args.corpus == "bootstrap":
            print(
                "\nRun RQ1 first: python scripts/rq1_run.py --execute",
                file=sys.stderr,
            )
        else:
            print(
                "\nRun RQ1 v2 first: python scripts/rq1_run.py --execute --corpus real",
                file=sys.stderr,
            )
        sys.exit(1)

    print(
        f"Scoring {len(rows)} eval row(s) × 3 arms (eval_queries router accuracy only)...",
        file=sys.stderr,
    )

    for arm_id, arm in arms.items():
        print(f"\n--- {arm_id}: {arm['label']} ---", file=sys.stderr)
        from arcs.router.classifier import clear_cache

        clear_cache()
        arm["eval_result"] = _score_subset(rows, model_dir=arm["model_dir"])
        acc = _eval_queries_acc(arm["eval_result"])
        n = (arm["eval_result"].get("metrics") or {}).get("n")
        print(f"  n={n}  accuracy={_fmt_acc(acc)}", file=sys.stderr)

    table = _build_table(arms)
    winner = _pick_winner(table)

    manifest: dict[str, Any] = {
        "kind": KIND,
        "name": "rq1-bis",
        "hypothesis": (
            "Router repair retraining (Run A all negatives vs Run B ROUTER-only) "
            "improves eval_queries routing accuracy over the pre-repair baseline."
        ),
        "corpus": args.corpus,
        "eval_queries": str(args.input),
        "n_queries": len(rows),
        "filters": {
            "ids": sorted(id_filter) if id_filter else None,
            "domains": sorted(domain_filter) if domain_filter else None,
            "limit": args.limit,
        },
        "arms": {
            arm_id: {
                "label": arm["label"],
                "description": arm["description"],
                "model_dir": str(arm["model_dir"]),
                "eval_queries_router_accuracy": _eval_queries_acc(arm["eval_result"]),
                "eval_result": arm["eval_result"],
            }
            for arm_id, arm in arms.items()
        },
        "table": table,
        "winner": winner,
        "winner_criteria": "highest eval_queries router accuracy vs arm0",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + "_repair_ablation"
    manifest_dir = config.EXPERIMENTS_DIR / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _print_table(table)
    print(f"\nWinner: {winner}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
