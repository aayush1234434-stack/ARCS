"""Isolated A/B for the LLM judge model.

Generates each answer+spec once (cached), then scores the same
(question, answer, spec) triples with multiple judge models so the judge model
is the only variable. Use to decide whether a stronger NVIDIA_JUDGE_MODEL is
worth adopting for domains the 8B judge over-fails (e.g. LEGAL).

    python scripts/judge_ab.py --domains LEGAL --limit 5 \
        --models meta/llama-3.1-8b-instruct meta/llama-3.3-70b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs import config, progress
from arcs.main import _get_components
from arcs.pipelines import resolve_pipeline
from arcs.verification import judge

EVAL_FILE = config.PROJECT_ROOT / "data" / "eval_queries.jsonl"
CACHE_DIR = config.ARTIFACTS_DIR / "judge_ab_cache"


def _load_rows(domains: set[str], limit: int | None) -> list[dict]:
    rows = []
    for line in EVAL_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if domains and row.get("expected_domain") not in domains:
            continue
        rows.append(row)
    return rows[:limit] if limit else rows


def _generate_triple(row: dict, components: dict) -> dict:
    """Answer + spec for one query, cached on disk so re-runs are free."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{row['id']}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    query = row["query"]
    pipeline = resolve_pipeline(row["expected_domain"], use_fallback=False)
    model = pipeline.resolve_model()
    answer = pipeline.specialist.run(query, model=model)["answer"]
    spec = components["spec_generator"].run(query)
    triple = {"id": row["id"], "query": query, "answer": answer, "spec": spec}
    cache_path.write_text(json.dumps(triple, indent=2))
    return triple


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B the LLM judge model.")
    parser.add_argument("--domains", default="LEGAL", help="comma-separated domains")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct"],
    )
    args = parser.parse_args()

    progress.set_verbose(False)
    domains = {d.strip().upper() for d in args.domains.split(",") if d.strip()}
    rows = _load_rows(domains, args.limit)
    components = _get_components()

    print(f"Generating {len(rows)} triple(s) for {sorted(domains)}...", file=sys.stderr)
    triples = []
    for row in rows:
        print(f"  gen {row['id']}", file=sys.stderr)
        triples.append(_generate_triple(row, components))

    results: dict[str, list] = {m: [] for m in args.models}
    for model in args.models:
        print(f"\n=== judge model: {model} ===", file=sys.stderr)
        for t in triples:
            start = time.perf_counter()
            try:
                res = judge.run(t["query"], t["answer"], t["spec"], model=model)
                verdict, score = res["verdict"], res["score"]
            except Exception as exc:  # noqa: BLE001
                verdict, score = f"ERROR({exc})", 0.0
            ms = int((time.perf_counter() - start) * 1000)
            results[model].append((t["id"], verdict, score))
            print(f"  {t['id']}: {verdict} score={score:.2f} ({ms}ms)", file=sys.stderr)

    print("\n=== summary (PASS count) ===")
    for model in args.models:
        passes = sum(1 for _, v, _ in results[model] if v == "PASS")
        print(f"  {model}: {passes}/{len(triples)} PASS")


if __name__ == "__main__":
    main()
