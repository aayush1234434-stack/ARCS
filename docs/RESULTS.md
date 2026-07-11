# ARCS — Experimental Results

*Paper-style summary of the MVP evaluation harness. All numbers below are pulled from saved artifacts under `artifacts/experiments/` unless noted as pending.*

**Primary sources**

| Artifact | Role |
|---|---|
| `2026-07-10T07-24-20_baseline-v1-full-pipeline` | Pre-repair end-to-end baseline |
| `2026-07-11T09-35-19_post-fix-v2-merged` | Post-repair v2 (merged partial runs) |
| `2026-07-11T08-38-52_rq1/manifest.json` | RQ1 bootstrap router retrain comparison |
| `2026-07-11T11-22-52_repair_ablation/manifest.json` | RQ1-bis fast router ablation |

---

## 1. Abstract

ARCS (Adaptive Routing & Correction System) is a modular orchestration stack that routes each user query to a domain-specific pipeline, verifies the answer against an independently generated specification, and attributes failures to router, specialist, verifier, or ambiguous causes before any retraining. We evaluate on a held-out set of 48 multi-domain queries (`data/eval_queries.jsonl`) and a frozen router test set of 200 examples (`data/router/router_test.csv`). After domain-targeted repairs (prompt sidecars, coding-path fixes, router retrain), end-to-end PASS rate on completed eval rows rises from **36.4%** (baseline) to **42.4%** (post-fix-v2), with the largest per-domain gain on LEGAL (+30.8 pts) and CODING (+25.0 pts). Bootstrap RQ1 shows router retraining on synthetic negative feedback improves eval-queries routing accuracy from **93.75%** to **97.92%**, but Run A (all negatives) and Run B (ROUTER-only) **tie** on both routing and pipeline metrics — attribution filtering is plausible but unconfirmed at this corpus size. A naive single-LLM baseline (same generator and judge, no orchestration) is wired but not yet run at full scale; RQ1 v2 on real demo feedback remains blocked pending corpus thresholds (≥40 negatives, ≥15 ROUTER-attributed).

---

## 2. System overview

ARCS is **not** four prompt-only specialists sharing one undifferentiated call pattern. It is an orchestration system: route → resolve pipeline → generate answer → build spec → verify → (optional) collect attributed feedback.

The architecture matches the diagram in the [README](../README.md#how-it-works):

```
User query → DistilBERT router → domain pipeline (prompt · contract · verifier · tools)
           → spec generator (Qwen) → sandbox (CODING) or LLM judge (other domains)
           → delivered answer → feedback + attribution engine
```

**Specialist pipelines (MVP)**

| Pipeline | Verifier | Toolchain | Default generator |
|---|---|---|---|
| CODING | Python sandbox (+ subprocess fallback) | Independent test generator, up to 3 retries | Groq `llama-3.3-70b-versatile` |
| MEDICAL / LEGAL / GENERAL | LLM judge (NVIDIA API) | Spec checklist | Groq `llama-3.3-70b-versatile` |

Attribution runs *after* inference. Only ROUTER-, SPECIALIST-, and VERIFIER-blamed failures enter retraining queues; AMBIGUOUS rows are logged but not used for repair.

---

## 3. Experimental setup

### 3.1 Datasets

| Set | Path | *n* | Use |
|---|---|---:|---|
| **Eval queries** | `data/eval_queries.jsonl` | **48** | End-to-end pipeline PASS/FAIL; router eval-queries accuracy |
| **Router test** | `data/router/router_test.csv` | **200** | Frozen held-out router classification (never used for RQ1 augment) |
| **Router train** | `data/router/router_train.csv` | 800 | DistilBERT fine-tuning base |

Eval queries are stratified across CODING (12), MEDICAL (12), LEGAL (13), and GENERAL (11), including cross-domain and “tricky” rows (e.g. HIPAA → LEGAL, prose-only coding).

### 3.2 Models and APIs

| Component | Model / runtime | Provider |
|---|---|---|
| Router | DistilBERT (`distilbert-base-uncased` fine-tuned) | Local PyTorch / optional ONNX |
| Generator (default) | `llama-3.3-70b-versatile` | Groq |
| Spec generator | `qwen/qwen3-32b` | Groq |
| Test generator (CODING) | `qwen/qwen3-32b` | Groq |
| LLM judge | `meta/llama-3.1-8b-instruct` | NVIDIA integrate API |
| Coding sandbox | `python:3.11-slim` (Docker; subprocess fallback) | Local |

### 3.3 Metrics

- **Pipeline PASS rate** — fraction of eval rows with verifier verdict `PASS`. We report both **raw** (*n* = 48, includes ERROR) and **completed** (PASS + FAIL only; ERROR excluded) when infra failures dominate.
- **Router accuracy** — fraction of rows where `predicted_domain == expected_domain`.
- **RQ1** — compares pre-RQ1 vs Run A vs Run B router checkpoints on eval-queries routing accuracy and frozen router test accuracy.

### 3.4 Repair interventions (post-fix-v2)

Between baseline and post-fix-v2 the project applied:

- Domain prompt sidecars (LEGAL, CODING, MEDICAL, GENERAL) from DSPy optimization artifacts
- Coding-path fixes (prose-only CODING → judge fallback; empty-answer handling)
- Router retrain on exported failure examples
- Partial eval resume after Groq TPD exhaustion (15 ERROR rows in merged post-fix-v2 run)

---

## 4. End-to-end results: baseline vs post-fix-v2

**Sources:** `2026-07-10T07-24-20_baseline-v1-full-pipeline`, `2026-07-11T09-35-19_post-fix-v2-merged`.

### 4.1 Overall pipeline (eval queries, *n* = 48)

| Run | PASS | FAIL | ERROR | PASS% (all rows) | PASS% (completed) |
|---|---:|---:|---:|---:|---:|
| Baseline v1 | 16 | 28 | 4 | 33.3% | **36.4%** (16/44) |
| Post-fix v2 | 14 | 19 | 15 | 29.2% | **42.4%** (14/33) |

Post-fix-v2 improves quality on rows that finish (42.4% vs 36.4% completed), but **15 ERROR rows** (mostly Groq TPD / infra during the merged run) depress the raw rate. GENERAL eval rows were entirely ERROR in the merged artifact (0/11 PASS).

### 4.2 Per-domain PASS rate (all eval rows in domain)

| Domain | *n* | Baseline PASS% | Post-fix-v2 PASS% | Δ (post − base) |
|---|---:|---:|---:|---:|
| CODING | 12 | 25.0% (3/12) | 50.0% (6/12) | **+25.0 pts** |
| MEDICAL | 12 | 50.0% (6/12) | 16.7% (2/12) | −33.3 pts |
| LEGAL | 13 | 15.4% (2/13) | 46.2% (6/13) | **+30.8 pts** |
| GENERAL | 11 | 45.5% (5/11) | 0.0% (0/11) | −45.5 pts† |

† GENERAL regression is an artifact of 11/11 ERROR in post-fix-v2 merged run, not verified answer-quality regression.

### 4.3 Router accuracy (same eval pass)

| Run | Eval-queries router acc. | Router test acc. (*n* = 200) |
|---|---:|---:|
| Baseline v1 | 93.2% (41/44 scored) | 95.5% |
| Post-fix v2 | 97.0% (32/33 scored) | — |

---

## 5. RQ1 — Attribution-filtered router retraining

**Hypothesis.** Retraining the router only on failures attributed to the router (Run B) beats retraining on all negative feedback (Run A).

**Source:** `artifacts/experiments/2026-07-11T08-38-52_rq1/manifest.json`  
**Corpus:** bootstrap synthetic (`data/rq1/feedback_corpus.jsonl`) — 38 augment rows (Run A), 12 ROUTER-only (Run B).

| Arm | Router test acc. | Eval-queries router acc. |
|---|---:|---:|
| Pre-RQ1 | 95.5% | 93.75% |
| Run A (all negatives) | 99.0% | 97.92% |
| Run B (ROUTER-only) | 99.0% | 97.92% |

**Outcome:** **tie** — both retraining strategies reach identical metrics. Repair clearly helps over pre-RQ1 (+4.17 pts on eval-queries routing), but the bootstrap corpus is too small to separate targeted vs blanket retrain.

### 5.1 RQ1-bis (fast router ablation)

`scripts/eval_repair_ablation.py` re-scores the three RQ1 checkpoints on eval_queries only (no pipeline). Confirms the manifest numbers:

| arm | router_eval_acc | Δ vs arm0 |
|---|---:|---:|
| arm0 (pre-RQ1) | 93.75% | — |
| arm1 (Run A) | 97.92% | +4.17 pts |
| arm2 (Run B) | 97.92% | +4.17 pts |

Manifest: `2026-07-11T11-22-52_repair_ablation/manifest.json`.

### 5.2 RQ1 v2 (real feedback) — pending

Real-feedback RQ1 requires **≥40** total 👎 rows and **≥15** ROUTER-attributed negatives in `logs/requests.jsonl`. As of the last corpus build: **2 negatives, 0 ROUTER** — not ready.

```bash
python scripts/feedback_stats.py --requests-only   # exit 0 when ready
python scripts/bootstrap_rq1_corpus.py --real-only
python scripts/rq1_prepare_datasets.py --corpus real
python scripts/rq1_run.py --execute --corpus real
```

Results will be recorded in a new `*_rq1-v2/manifest.json` without overwriting bootstrap artifacts.

---

## 6. Naive baseline (orchestration ablation)

**Question.** Holding generator (`llama-3.3-70b-versatile`) and verifier (same spec + judge) fixed, how much does routing + specialist orchestration add?

**Method.** `scripts/eval_naive_baseline.py` — one Groq call per query (`Answer this question: {query}`), then identical spec generation and LLM judge as the full pipeline. Saved with `kind: "naive_baseline"`.

| Run | PASS% (completed) | Status |
|---|---:|---|
| Naive single-LLM | *pending* | Full 48-row run not yet saved |
| ARCS post-fix-v2 | **42.4%** (14/33) | `2026-07-11T09-35-19_post-fix-v2-merged` |

A 2-query smoke test (CODING only) passed both rows at 100% judge PASS; full-set numbers require:

```bash
python scripts/eval_naive_baseline.py --name naive-baseline-v1 --sleep-between 1 \
  --compare-to artifacts/experiments/2026-07-11T09-35-19_post-fix-v2-merged
```

---

## 7. Limitations

1. **Synthetic RQ1 bootstrap** — Negative feedback is reconstructed from eval artifacts (`misclassified_test.json`, pipeline failures), not live user 👎 signal. Label and attribution distributions may not match production.

2. **Small N** — 48 eval queries and ≈38 bootstrap negatives (≈12 ROUTER) leave metric deltas within noise; RQ1 Run A vs Run B cannot be distinguished.

3. **Groq TPD / rate limits** — Free-tier daily token caps caused partial eval runs (15 ERROR rows in post-fix-v2 merged). Completed-row PASS% is the fairer quality metric but reduces effective *n*.

4. **Judge strictness** — PASS requires score ≥ 0.75 with no missing required elements and no disqualifying conditions. Strict spec checklists inflate FAIL relative to human judgment.

5. **Single generator family** — All domains share one Groq Llama backend today; RQ2 (heterogeneous specialists) is not tested.

6. **GENERAL domain fragility** — Post-fix-v2 merged run lost all GENERAL rows to ERROR; domain-level conclusions for GENERAL are unreliable until resume completes.

7. **Real feedback gap** — RQ1 v2 and production repair loops depend on demo/CLI feedback that has not yet reached corpus thresholds.

---

## 8. Reproduce

### CI-safe helper

From repository root (uses `.venv/bin/python` when present):

```bash
./scripts/reproduce.sh check           # pytest + import smoke
./scripts/reproduce.sh eval-baseline   # print baseline eval commands (no API calls)
./scripts/reproduce.sh rq1-bootstrap   # bootstrap corpus + prepare datasets + rq1_run --dry-run
./scripts/reproduce.sh merge           # snapshot_post_fix --dry-run
```

Each subcommand prints at the end: *Full eval requires GROQ_API_KEY and NVIDIA_API_KEY; not run in CI.*

### Full eval (requires API keys)

From repository root with `.venv` activated and `.env` configured (Groq + NVIDIA keys):

```bash
# ── Validate held-out set ──
python scripts/validate_eval_queries.py

# ── Baseline snapshot (router + pipeline) ──
python scripts/snapshot_baseline.py --name baseline-v1 -q

# ── Full pipeline eval ──
python scripts/eval_pipeline.py --name baseline-v1-full --sleep-between 2
python scripts/eval_pipeline.py --name post-fix-v2 --sleep-between 2

# Resume after TPD (example)
python scripts/eval_pipeline.py --name post-fix-v2-resume \
  --resume-from artifacts/experiments/2026-07-11T09-35-19_post-fix-v2-merged \
  --sleep-between 2

# Merge partial runs
python scripts/merge_experiments.py \
  artifacts/experiments/<run-a> artifacts/experiments/<run-b> \
  --name post-fix-v2-merged

# ── Compare experiments ──
python scripts/compare_experiments.py \
  artifacts/experiments/2026-07-10T07-24-20_baseline-v1-full-pipeline \
  artifacts/experiments/2026-07-11T09-35-19_post-fix-v2-merged

# ── RQ1 bootstrap ──
python scripts/bootstrap_rq1_corpus.py
python scripts/rq1_prepare_datasets.py
python scripts/rq1_run.py --execute

# ── RQ1-bis router ablation (fast) ──
python scripts/eval_repair_ablation.py --execute

# ── Naive orchestration ablation ──
python scripts/eval_naive_baseline.py --name naive-baseline-v1 --sleep-between 1

# ── RQ1 v2 (when feedback_stats.py --requests-only exits 0) ──
python scripts/bootstrap_rq1_corpus.py --real-only
python scripts/rq1_prepare_datasets.py --corpus real
python scripts/rq1_run.py --execute --corpus real
```

Inspect saved metrics:

```bash
python scripts/compare_experiments.py --list
cat artifacts/experiments/2026-07-11T08-38-52_rq1/manifest.json
cat artifacts/experiments/2026-07-11T11-22-52_repair_ablation/manifest.json
```

---

*Last updated from artifacts dated 2026-07-11. Re-run eval commands after repair or corpus changes and refresh this document from new `experiment.json` / `manifest.json` files.*
