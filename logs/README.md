# ARCS runtime logs

Gitignored directory for live feedback, eval-derived bootstrap rows, and attributed repair queues. Created automatically by the demo, CLI, and export scripts.

---

## `requests.jsonl`

**Primary log** — one JSON object per pipeline run from the demo UI or CLI (`python -m arcs.main`, `scripts/run_demo.py` → `/api/query`).

Typical fields:

| Field | Meaning |
|---|---|
| `query_id` | UUID for this run — use for 👎 feedback and cross-referencing errors |
| `timestamp` | UTC ISO-8601 |
| `status` | `PASS`, `FAIL`, `ERROR`, or `UNKNOWN` |
| `query` | User text |
| `route` | Router output (`domain`, `confidence`, …) |
| `specialist` | Domain answer payload |
| `verification` | Judge or sandbox result |
| `user_feedback` | `POSITIVE` / `NEGATIVE` when collected |
| `attribution` | Blame component after NEGATIVE feedback |
| `error` | Present when the pipeline aborted |
| `error_class` | Coarse bucket when `error` is set: `rate_limit`, `judge_parse`, `sandbox`, `empty_code`, `unknown` |

**Never overwritten** by eval export. Append-only.

Generate / refresh repair queues:

```bash
python scripts/extract_queues.py
```

---

## `eval_failures.jsonl`

**Optional bootstrap log** — synthetic NEGATIVE records from eval **FAIL** rows (not ERROR), produced by `scripts/export_eval_failures.py`.

Use when live 👎 feedback is sparse but you want Phase 1 repair queues populated for demos. Same general shape as `requests.jsonl` (attribution, verification, etc.) but sourced from saved experiments under `artifacts/experiments/`.

```bash
python scripts/export_eval_failures.py --dry-run
python scripts/export_eval_failures.py
```

Does **not** modify `requests.jsonl`. After export, run `extract_queues.py` to refresh `queues/` from **both** files (unless `--requests-only`).

---

## `queues/`

Attributed **NEGATIVE** rows split by blame component for Phase 1 repair. Produced by `scripts/extract_queues.py` (or `scripts/repair.py --dry-run`).

| File | Repair path |
|---|---|
| `router_queue.jsonl` | Label → `retrain_router.py` |
| `specialist_queue.jsonl` | `optimize_{medical,coding,legal,general}.py` |
| `verifier_queue.jsonl` | `optimize_judge.py` |
| `ambiguous_queue.jsonl` | Human review only — do not train |

Each line is a full log record plus queue metadata. Rows with `user_feedback != NEGATIVE` or missing attribution are skipped.

Default input merge: `requests.jsonl` + `eval_failures.jsonl` (when present).

---

## Debugging production errors

When `status == ERROR` in `requests.jsonl`, check:

1. **`query_id`** — always assigned at pipeline start (even on failure)
2. **`error_class`** — quick filter:
   - `rate_limit` — Groq/NVIDIA 429; TPD may need resume next day
   - `judge_parse` — judge response not valid JSON
   - `sandbox` — coding sandbox execution failure
   - `empty_code` — coding answer had no runnable fenced block
   - `unknown` — everything else

Eval runs (`scripts/eval_pipeline.py`) mirror `error_class` on ERROR rows and print a summary breakdown at the end.
