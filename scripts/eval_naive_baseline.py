"""CLI: run the held-out eval set through a NAIVE single-LLM baseline.

The naive baseline deliberately skips ARCS orchestration — no router, no
domain specialist pipeline, no sandbox. For each eval query it makes a single
Groq call ("Answer this question: {query}") with the same generator model the
ARCS specialists use, then verifies the answer with the SAME spec generator +
LLM judge as the full pipeline. This isolates the effect of orchestration:
generator and verifier are held constant, only the routing/specialist layer is
removed, so a PASS-rate gap is attributable to orchestration.

Experiments are saved under artifacts/experiments/ with kind="naive_baseline"
so they never collide with pipeline runs and can be diffed against a post-fix
ARCS experiment.

Usage:
    python scripts/eval_naive_baseline.py --dry-run
    python scripts/eval_naive_baseline.py --name naive-baseline-v1
    python scripts/eval_naive_baseline.py --limit 5 --domains CODING,MEDICAL
    python scripts/eval_naive_baseline.py --name naive-baseline-v1 \
        --compare-to artifacts/experiments/<post-fix-v2-merged>
    python scripts/eval_naive_baseline.py --compare-only artifacts/experiments/<naive-run> \
        --baseline-experiment artifacts/experiments/<post-fix-v2-merged>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.clients.rate_limit import is_groq_tpd_exhausted
from arcs.eval.compare import format_orchestration_comparison, pass_stats
from arcs.eval.experiments import latest_experiment, load_experiment, save_experiment
from arcs.eval.metrics import (
    VALID_DOMAINS,
    aggregate_experiment,
    pipeline_summary,
    router_accuracy,
)

DEFAULT_INPUT = config.DATA_DIR / "eval_queries.jsonl"
DOMAIN_SET = frozenset(VALID_DOMAINS)
NAIVE_MODEL = config.DEFAULT_GENERATOR_MODEL
NAIVE_PROMPT = "Answer this question: {query}"
KIND = "naive_baseline"

# Exit code when Groq's daily token limit (TPD) is exhausted (partial saved).
EXIT_TPD_EXHAUSTED = 2


def _load_rows(path: Path) -> list[dict[str, Any]]:
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
            if not isinstance(query, str) or not query.strip():
                print(
                    f"Warning: skipping line {line_number} — missing query",
                    file=sys.stderr,
                )
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


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _naive_answer(query: str, *, model: str) -> str:
    """Single Groq call — no router, no specialist pipeline, no tools."""
    from arcs.clients.groq import get_client

    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": NAIVE_PROMPT.format(query=query)}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _verdict_to_status(verification: dict[str, Any]) -> str:
    verdict = str(verification.get("verdict") or "").strip().upper()
    if verdict == "PASS":
        return "PASS"
    if verdict == "FAIL":
        return "FAIL"
    return "UNKNOWN"


def _run_one(row: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Run one query through the naive baseline: answer -> spec -> judge."""
    from arcs.verification import judge, spec_generator

    query = str(row["query"]).strip()
    expected = _normalize_domain(row.get("expected_domain"))
    result: dict[str, Any] = {
        "id": row.get("id"),
        "query": query,
        "expected_domain": expected,
        # Naive baseline does no routing — predicted_domain is always None so it
        # is excluded from router accuracy but still counted for pipeline PASS%.
        "predicted_domain": None,
        "router_confidence": None,
        "pipeline_id": "NAIVE",
        "verifier": "llm_judge",
        "error": None,
    }
    timing: dict[str, int] = {}
    started = time.perf_counter()

    answer_start = time.perf_counter()
    answer = _naive_answer(query, model=model).strip()
    timing["answer_ms"] = _elapsed_ms(answer_start)
    result["answer"] = answer

    if not answer:
        result["status"] = "FAIL"
        result["verification"] = {
            "verdict": "FAIL",
            "score": 0.0,
            "explanation": "naive model returned empty answer",
        }
        timing["total_ms"] = _elapsed_ms(started)
        result["timing"] = timing
        return result

    spec_start = time.perf_counter()
    specification = spec_generator.run(query)
    timing["specification_ms"] = _elapsed_ms(spec_start)

    verify_start = time.perf_counter()
    verification = judge.run(question=query, answer=answer, specification=specification)
    timing["verification_ms"] = _elapsed_ms(verify_start)
    timing["total_ms"] = _elapsed_ms(started)

    score = verification.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None

    result["specification"] = specification
    result["verification"] = {
        "verdict": verification.get("verdict"),
        "score": score_f,
        "explanation": verification.get("explanation", ""),
    }
    result["status"] = _verdict_to_status(verification)
    result["timing"] = timing
    return result


def _error_row(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "query": str(row.get("query") or "").strip(),
        "expected_domain": _normalize_domain(row.get("expected_domain")),
        "predicted_domain": None,
        "router_confidence": None,
        "pipeline_id": "NAIVE",
        "verifier": "llm_judge",
        "status": "ERROR",
        "error": error,
        "verification": {"verdict": None, "score": None},
        "timing": {},
    }


def run_eval(
    rows: list[dict[str, Any]],
    *,
    model: str,
    dry_run: bool,
    sleep_between: float = 0.0,
) -> tuple[list[dict[str, Any]], bool]:
    """Run naive-baseline eval rows. Returns (results, tpd_exhausted)."""
    results: list[dict[str, Any]] = []
    total = len(rows)
    tpd_exhausted = False

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id") or f"row-{index}"
        expected = _normalize_domain(row.get("expected_domain"))

        if dry_run:
            results.append(
                {
                    "id": row.get("id"),
                    "query": str(row.get("query") or "").strip(),
                    "expected_domain": expected,
                    "status": "PLANNED",
                    "predicted_domain": None,
                    "error": None,
                    "timing": {},
                }
            )
            print(
                f"[{index}/{total}] {row_id}  planned  expected={expected or '?'}",
                file=sys.stderr,
            )
            continue

        try:
            result = _run_one(row, model=model)
        except Exception as exc:  # noqa: BLE001 — keep the eval going
            result = _error_row(row, str(exc))
            print(f"  ERROR on {row_id}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if is_groq_tpd_exhausted(exc):
                results.append(result)
                tpd_exhausted = True
                print(
                    "\nGroq daily token limit (TPD) exhausted — stopping early.",
                    file=sys.stderr,
                )
                break

        results.append(result)
        status = result.get("status")
        score = (result.get("verification") or {}).get("score")
        score_note = f"  score={score:.2f}" if isinstance(score, (int, float)) else ""
        total_ms = (result.get("timing") or {}).get("total_ms")
        latency = f"  {int(total_ms)}ms" if isinstance(total_ms, (int, float)) else ""
        print(
            f"[{index}/{total}] {row_id}  status={status}  "
            f"expected={expected or '?'}{score_note}{latency}",
            file=sys.stderr,
        )

        if sleep_between > 0 and index < total:
            time.sleep(sleep_between)

    return results, tpd_exhausted


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:5.1f}%"


def _print_simple_delta(naive: dict[str, Any], arcs_exp: dict[str, Any]) -> None:
    """Print compact PASS rate table with delta."""
    naive_stats = pass_stats(naive)
    arcs_stats = pass_stats(arcs_exp)
    n_all = naive_stats.get("n") or arcs_stats.get("n")
    print("\n=== Naive vs ARCS (orchestration ablation) ===", file=sys.stderr)
    print(f"{'System':<22} {'PASS rate (48)':>16}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    naive_pct = naive_stats["pass_pct"]
    arcs_pct = arcs_stats["pass_pct"]
    naive_label = f"{naive_stats['pass']}/{naive_stats['completed']}"
    arcs_label = f"{arcs_stats['pass']}/{arcs_stats['completed']}"
    print(
        f"{'Naive LLM + judge':<22} "
        f"{_fmt_pct(naive_pct) if naive_pct is not None else 'TBD':>16}",
        file=sys.stderr,
    )
    print(
        f"{'ARCS post-fix':<22} "
        f"{_fmt_pct(arcs_pct) if arcs_pct is not None else 'TBD':>16}",
        file=sys.stderr,
    )
    if naive_pct is not None and arcs_pct is not None:
        delta = arcs_pct - naive_pct
        print(
            f"\nARCS − naive: {delta:+.1f} pts ({arcs_label} vs {naive_label} completed)",
            file=sys.stderr,
        )
    if n_all:
        print(f"(eval corpus n={n_all})", file=sys.stderr)


def _print_comparison(naive: dict[str, Any], arcs_exp: dict[str, Any]) -> None:
    _print_simple_delta(naive, arcs_exp)
    print("\n=== " + format_orchestration_comparison(naive, arcs_exp).rstrip(), file=sys.stderr)


def _run_compare_only(naive_path: Path, baseline_path: Path) -> None:
    naive_exp = load_experiment(naive_path)
    arcs_exp = load_experiment(baseline_path)
    _print_comparison(naive_exp, arcs_exp)


def _resolve_experiment_path(raw: str) -> Path | None:
    """Resolve an experiment directory or ``experiment.json`` path."""
    path = Path(raw)
    if path.is_file() and path.name == "experiment.json":
        return path.parent
    if path.is_dir() and (path / "experiment.json").is_file():
        return path
    if path.exists():
        return path
    candidate = config.EXPERIMENTS_DIR / raw
    if candidate.is_dir() and (candidate / "experiment.json").is_file():
        return candidate
    return None


def _resolve_compare_to(raw: str | None) -> Path | None:
    """Resolve the ARCS experiment to compare against.

    ``raw`` may be a path, a run_id under artifacts/experiments/, or None
    (auto-pick the newest post-fix experiment, else the newest experiment).
    """
    if raw is not None:
        resolved = _resolve_experiment_path(raw)
        if resolved is not None:
            return resolved
        print(f"Warning: experiment not found: {raw}", file=sys.stderr)
        return None

    root = config.EXPERIMENTS_DIR
    if not root.exists():
        return None
    post_fix = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir()
            and (p / "experiment.json").exists()
            and "post-fix" in p.name
        ),
        key=lambda p: p.name,
    )
    if post_fix:
        return post_fix[-1]
    return latest_experiment(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run data/eval_queries.jsonl through a naive single-LLM baseline "
            "(no router/specialist), verified with the same spec + judge as ARCS."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Eval JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run at most N rows after filters",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated eval ids to include (e.g. eval-001,eval-002)",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated expected domains (e.g. CODING,MEDICAL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and counts; do not call any model or write artifacts",
    )
    parser.add_argument(
        "--name",
        default="naive-baseline-v1",
        help="Experiment name (default: naive-baseline-v1)",
    )
    parser.add_argument(
        "--model",
        default=NAIVE_MODEL,
        help=f"Groq generator model for the naive call (default: {NAIVE_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Experiments root (default: {config.EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        metavar="N",
        help="Seconds to sleep between rows to ease rate limits (default: 0)",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        metavar="PATH",
        help=(
            "ARCS experiment (path or run_id) to compare PASS%% against. "
            "Default: newest post-fix experiment under artifacts/experiments/. "
            "Alias: --baseline-experiment."
        ),
    )
    parser.add_argument(
        "--baseline-experiment",
        default=None,
        metavar="PATH",
        help=(
            "ARCS post-fix experiment to compare against (same as --compare-to). "
            "Use with --compare-only for a dry-run delta on saved artifacts."
        ),
    )
    parser.add_argument(
        "--compare-only",
        default=None,
        metavar="PATH",
        help=(
            "Skip eval; load a saved naive_baseline experiment and print PASS%% "
            "delta vs --baseline-experiment (no API calls)."
        ),
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip the naive-vs-ARCS PASS%% comparison table",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write experiment artifacts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full experiment dict to stdout",
    )
    args = parser.parse_args()

    baseline_raw = args.baseline_experiment or args.compare_to

    if args.compare_only is not None:
        naive_path = _resolve_experiment_path(args.compare_only)
        if naive_path is None:
            print(f"Error: --compare-only not found: {args.compare_only}", file=sys.stderr)
            sys.exit(1)
        baseline_path = _resolve_compare_to(baseline_raw)
        if baseline_path is None:
            print(
                "Error: --baseline-experiment (or --compare-to) is required with --compare-only",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            _run_compare_only(naive_path, baseline_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"Error: could not load experiment: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.input.exists():
        print(f"Error: eval file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.limit is not None and args.limit < 0:
        print("Error: --limit must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.sleep_between < 0:
        print("Error: --sleep-between must be >= 0", file=sys.stderr)
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

    rows = _load_rows(args.input)
    rows = _filter_rows(rows, ids=id_filter, domains=domain_filter, limit=args.limit)
    if not rows:
        print("Error: no eval rows matched filters", file=sys.stderr)
        sys.exit(1)

    planned_counts = Counter(
        _normalize_domain(row.get("expected_domain")) or "UNKNOWN" for row in rows
    )
    print(
        f"Loaded {len(rows)} eval row(s) from {args.input}  [naive baseline]"
        + (" [dry-run]" if args.dry_run else "")
        + (" [no-save]" if args.no_save else ""),
        file=sys.stderr,
    )
    print(f"  model: {args.model}", file=sys.stderr)
    for domain in VALID_DOMAINS:
        print(f"  {domain}: {planned_counts.get(domain, 0)}", file=sys.stderr)

    results, tpd_exhausted = run_eval(
        rows,
        model=args.model,
        dry_run=args.dry_run,
        sleep_between=args.sleep_between,
    )

    if args.dry_run:
        print(
            f"\n(dry-run: planned {len(rows)} row(s); no model calls, no artifacts)",
            file=sys.stderr,
        )
        return

    router = router_accuracy(results)
    pipeline = pipeline_summary(results)
    experiment = aggregate_experiment(
        args.name,
        router=router,
        pipeline=pipeline,
        meta={
            "input": str(args.input),
            "n_rows": len(results),
            "model": args.model,
            "naive_prompt": NAIVE_PROMPT,
            "rows": results,
        },
    )
    experiment["kind"] = KIND

    saved_to: Path | None = None
    if not args.no_save:
        saved_to = save_experiment(experiment, name=args.name, output_dir=args.output_dir)
        experiment = load_experiment(saved_to)
        experiment["kind"] = KIND

    stats = pass_stats(experiment)
    print("\n=== Naive baseline summary ===", file=sys.stderr)
    print(f"name:      {experiment.get('name')}", file=sys.stderr)
    print(f"kind:      {KIND}", file=sys.stderr)
    print(
        f"pipeline:  n={stats['n']}  PASS={stats['pass']}  FAIL={stats['fail']}  "
        f"ERROR={stats['error']}  PASS%(completed)={_fmt_pct(stats['pass_pct'])}",
        file=sys.stderr,
    )
    if saved_to is not None:
        print(f"saved:     {saved_to}", file=sys.stderr)

    if not args.no_compare:
        compare_path = _resolve_compare_to(baseline_raw)
        if compare_path is not None:
            try:
                arcs_exp = load_experiment(compare_path)
                _print_comparison(experiment, arcs_exp)
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"Warning: could not load --compare-to: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(experiment, indent=2, default=str))

    if tpd_exhausted:
        sys.exit(EXIT_TPD_EXHAUSTED)


if __name__ == "__main__":
    main()
