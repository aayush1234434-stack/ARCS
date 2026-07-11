"""Bootstrap the repair demo from eval FAILs when live feedback is sparse.

Reads a merged/post-fix pipeline experiment, turns every FAIL row (not ERROR)
into a NEGATIVE, log-shaped feedback record, runs the real attribution engine on
it, and appends the results to logs/eval_failures.jsonl (never touching
logs/requests.jsonl). It then refreshes logs/queues/*.jsonl so the Phase 1 repair
tools have something to work on even before real 👎 feedback accumulates.

Usage:
    python scripts/export_eval_failures.py                 # default merged experiment
    python scripts/export_eval_failures.py --experiment PATH
    python scripts/export_eval_failures.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config
from arcs.eval.experiments import latest_experiment, load_experiment
from arcs.post.attribution import attribute
from arcs.post.queues import COMPONENTS, extract_queues, format_summary, queue_counts

EVAL_FAILURES_LOG = config.LOGS_DIR / "eval_failures.jsonl"
REQUESTS_LOG = config.LOGS_DIR / "requests.jsonl"
QUEUES_DIR = config.LOGS_DIR / "queues"


def _default_experiment() -> Path | None:
    """Prefer the newest *post-fix-v2-merged* run, else the newest experiment."""
    root = config.EXPERIMENTS_DIR
    if root.exists():
        merged = [
            p
            for p in root.iterdir()
            if p.is_dir()
            and p.name.endswith("_post-fix-v2-merged")
            and (p / "experiment.json").exists()
        ]
        if merged:
            return max(merged, key=lambda p: p.stat().st_mtime)
    return latest_experiment()


def _reconstruct_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a verification block including verification_type for attribution.

    Eval rows store only verdict/score; attribution rule 1 (sandbox failure)
    needs verification_type, which we recover from the row's ``verifier`` field.
    """
    verification = dict(row.get("verification") or {})
    verifier = str(row.get("verifier") or "").lower()
    if "verification_type" not in verification:
        if verifier == "sandbox":
            verification["verification_type"] = "SANDBOX"
        elif verifier in {"llm_judge", "judge"}:
            verification["verification_type"] = "LLM_JUDGE"
    return verification


def _minimal_specification(query: str) -> dict[str, Any]:
    """Query-derived spec when eval rows lack a persisted specification."""
    text = query.strip()
    return {
        "intent": text,
        "required_elements": [
            f"Directly address: {text[:240]}",
            "Provide a correct, complete answer for the question asked",
        ],
        "correctness_criteria": [
            "Answer stays on topic and covers the core request",
            "No materially incorrect or misleading claims",
        ],
        "disqualifying_conditions": [
            "Omits information essential to answering the question",
        ],
        "scope": "Stay within the scope of the user's question.",
        "source": "eval_export_minimal",
    }


def _extract_specification(row: dict[str, Any]) -> dict[str, Any]:
    specification = row.get("specification")
    if isinstance(specification, dict) and specification.get("required_elements"):
        return specification
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return _minimal_specification(query)
    return {}


def _extract_test_cases(row: dict[str, Any]) -> list[Any]:
    for container_key in ("specialist", "tooling", None):
        container = row if container_key is None else row.get(container_key)
        if not isinstance(container, dict):
            continue
        tests = container.get("test_cases") or container.get("tests")
        if isinstance(tests, list) and tests:
            return tests
    top = row.get("test_cases")
    if isinstance(top, list) and top:
        return top
    return []


def _build_specialist(row: dict[str, Any]) -> dict[str, Any]:
    specialist = dict(row.get("specialist") or {})
    answer = specialist.get("answer") or row.get("response") or specialist.get("answer_preview")
    if isinstance(answer, str) and answer.strip():
        specialist["answer"] = answer.strip()
    test_cases = _extract_test_cases(row)
    if test_cases:
        specialist["test_cases"] = test_cases
    pipeline_id = row.get("pipeline_id") or specialist.get("pipeline_id")
    if pipeline_id:
        specialist["pipeline_id"] = pipeline_id
    return specialist


def _answer_preview(row: dict[str, Any]) -> str:
    """Eval rows do not persist the answer text; use what is available."""
    specialist = row.get("specialist")
    if isinstance(specialist, dict):
        answer = specialist.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()[:280]
    return ""


def _build_record(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected_domain")
    verification = _reconstruct_verification(row)
    route = {
        "domain": row.get("predicted_domain"),
        "confidence": row.get("router_confidence"),
        "use_fallback": row.get("use_fallback"),
    }
    specification = _extract_specification(row)
    specialist = _build_specialist(row)
    if not specialist.get("answer"):
        preview = _answer_preview(row)
        if preview:
            specialist["answer_preview"] = preview
    record: dict[str, Any] = {
        "query_id": row.get("id"),
        "query": row.get("query"),
        "route": route,
        "pipeline": {
            "pipeline_id": row.get("pipeline_id"),
            "verifier": row.get("verifier"),
        },
        "specialist": specialist,
        "specification": specification,
        "verification": verification,
        "status": "FAIL",
        "user_feedback": "NEGATIVE",
        "source": "eval_export",
        "expected_domain": expected,
        "correct_domain": expected,
    }
    # Attribution reads route / verification / user_feedback / specialist.
    record["attribution"] = attribute(record)
    return record


def _fail_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (experiment.get("meta") or {}).get("rows") or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").upper() != "FAIL":
            continue  # skip ERROR / PASS / UNKNOWN
        if row.get("error"):
            continue
        out.append(row)
    return out


def _counts_by_component(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {component: 0 for component in COMPONENTS}
    for record in records:
        component = (record.get("attribution") or {}).get("component")
        if component in counts:
            counts[component] += 1
    return counts


def _dedupe_key(record: dict[str, Any]) -> Any:
    """Identity for a feedback record: its query_id, else the query text."""
    return record.get("query_id") or record.get("query")


def _upsert_jsonl(path: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Merge ``records`` into ``path`` keyed by query_id (idempotent re-runs).

    Existing rows are preserved; a new record with the same key overwrites the
    old one. Returns (added, updated) counts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = _dedupe_key(row)
            if key not in existing:
                order.append(key)
            existing[key] = row

    added = updated = 0
    for record in records:
        key = _dedupe_key(record)
        if key in existing:
            updated += 1
        else:
            added += 1
            order.append(key)
        existing[key] = record

    with path.open("w", encoding="utf-8") as fh:
        for key in order:
            fh.write(json.dumps(existing[key], ensure_ascii=False) + "\n")
    return added, updated


def _refresh_queues() -> dict[str, int]:
    """Rebuild queues from requests.jsonl + eval_failures.jsonl (combined).

    Combining preserves any real feedback already logged instead of clobbering
    the queues with only the eval-derived records.
    """
    sources = [p for p in (REQUESTS_LOG, EVAL_FAILURES_LOG) if p.exists()]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for source in sources:
            text = source.read_text(encoding="utf-8")
            tmp.write(text)
            if text and not text.endswith("\n"):
                tmp.write("\n")
        tmp_path = Path(tmp.name)
    try:
        queues = extract_queues(input_path=tmp_path, output_dir=QUEUES_DIR)
    finally:
        tmp_path.unlink(missing_ok=True)
    return queue_counts(queues)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export eval FAIL rows as attributed NEGATIVE feedback into "
            "logs/eval_failures.jsonl and refresh the repair queues."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Pipeline experiment dir or experiment.json (default: newest merged).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print attribution counts only; write nothing.",
    )
    args = parser.parse_args()

    experiment_path = args.experiment or _default_experiment()
    if experiment_path is None:
        print("Error: no experiment found; pass --experiment PATH.", file=sys.stderr)
        sys.exit(1)

    try:
        experiment = load_experiment(experiment_path)
    except (FileNotFoundError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: could not load experiment: {exc}", file=sys.stderr)
        sys.exit(1)

    fail_rows = _fail_rows(experiment)
    if not fail_rows:
        print(f"No FAIL rows in {experiment_path}. Nothing to export.", file=sys.stderr)
        return

    records = [_build_record(row) for row in fail_rows]
    counts = _counts_by_component(records)

    print(f"Experiment: {experiment_path}", file=sys.stderr)
    print(f"FAIL rows exported: {len(records)}", file=sys.stderr)
    print(f"  attribution: {format_summary(counts)}", file=sys.stderr)

    if args.dry_run:
        print("(dry-run: logs/eval_failures.jsonl and queues unchanged)", file=sys.stderr)
        return

    added, updated = _upsert_jsonl(EVAL_FAILURES_LOG, records)
    print(
        f"Wrote {EVAL_FAILURES_LOG} ({added} new, {updated} updated)",
        file=sys.stderr,
    )

    queue_totals = _refresh_queues()
    print(f"Refreshed queues in {QUEUES_DIR}", file=sys.stderr)
    print(f"  queue totals: {format_summary(queue_totals)}", file=sys.stderr)


if __name__ == "__main__":
    main()
