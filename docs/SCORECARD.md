# ARCS — Project Scorecard

*Honest self-assessment for portfolio / thesis committee. Scores use saved artifacts as of **2026-07-11** (`post-fix-v2-merged`, RQ1 manifest). Not a marketing doc.*

**Primary evidence**

| Artifact | What it proves |
|---|---|
| `artifacts/experiments/2026-07-10T07-24-20_baseline-v1-full-pipeline` | Pre-repair end-to-end baseline |
| `artifacts/experiments/2026-07-11T12-36-52_post-fix-v2-merged` | Post-fix **FINAL** (48/48 rows, 0 ERROR after domain runs + resume merge) |
| `artifacts/experiments/2026-07-11T08-38-52_rq1/manifest.json` | RQ1 bootstrap (Run A vs Run B) |
| `artifacts/experiments/2026-07-11T11-22-52_repair_ablation/manifest.json` | RQ1-bis fast router ablation |

**Headline numbers (eval queries, *n* = 48)**

| Metric | Baseline | Post-fix FINAL | Δ |
|---|---:|---:|---:|
| PASS (completed) | 16/44 → **36.4%** | 20/48 → **41.7%** | **+5.3 pp** |
| ERROR | 4 | 0 | −4 |
| Eval routing (RQ1 pre) | 93.75% | 97.92% (Run A = Run B) | +4.17 pp |

Per-domain PASS (all rows in domain): CODING **+17 pp**, LEGAL **+31 pp**, MEDICAL **−8 pp**, GENERAL **−9 pp** (see [README](../README.md#results-baseline-vs-post-fix)).

---

## Summary table

| Dimension | Score | Status |
|---|---:|---|
| Architecture | **8 / 10** | Strong MVP design; RQ2 heterogeneity not built |
| Implementation | **7 / 10** | End-to-end works; API fragility and partial uncommitted work |
| Eval rigor | **7 / 10** | Good harness; small *n*, merge stitching, no full CI eval |
| Research | **6 / 10** | Clear RQs; RQ1 inconclusive, naive baseline not run at scale |
| Production readiness | **5 / 10** | Skeleton only (Docker, ONNX, smoke) |
| Documentation | **8 / 10** | Extensive; committee one-pager now exists |

**Weighted thesis readiness:** ~**6.8 / 10** — credible **systems + methods** contribution; **empirical claims** need one more eval cycle and real-feedback RQ1.

---

## 1. Architecture — **8 / 10** · **done** (MVP)

### Evidence

- **Attribution before retraining** — failures sorted into ROUTER / SPECIALIST / VERIFIER / AMBIGUOUS queues before any repair (`arcs/post/attribution.py`, `scripts/repair.py`).
- **Pipeline = prompt + contract + verifier + tools**, not “four chat prompts.” CODING uses sandbox + retries; other domains use independent spec generator (Qwen) + LLM judge (NVIDIA).
- **Modular orchestrator** (`arcs/main.py`) with lazy-loaded components, domain registry, judge fallback for prose-only coding (`eval-042`).
- **Repair closed loop** wired: demo/CLI → `logs/requests.jsonl` → `extract_queues.py` → component-specific repair scripts.

### What would make it **9 / 10**

- Heterogeneous specialist backends (RQ2) behind stable pipeline interfaces.
- Explicit plugin boundary for verifiers (sandbox vs judge vs future tools) with versioned contracts.
- Event/outbox pattern for async repair jobs instead of batch scripts only.

### Phase map

| Phase | Item | Status |
|---|---|---|
| **A** | Repair loop architecture + queue extraction | **done** |
| **B** | RQ1/RQ1-bis experiment structure (isolated router eval) | **done** |
| **C** | ONNX router path (`ARCS_ROUTER_BACKEND`) | **done** |
| **D** | — | — |

---

## 2. Implementation — **7 / 10** · **mostly done**

### Evidence

- **90 pytest tests** (`tests/`, including `smoke_imports.py`, router backend, eval merge/resume, RQ1 v2 gates).
- Full pipeline runs locally: router → specialist → spec → verify → log; demo FastAPI wraps same path.
- **Eval resume** after Groq TPD (`eval_pipeline.py` exit code 2 + `--resume-from`).
- **Production debuggability:** `error_class` on pipeline failures (`rate_limit`, `judge_parse`, `sandbox`, `empty_code`, `unknown`); `query_id` preserved in logs.
- **Groq + NVIDIA** integration with rate-limit detection; judge JSON parse degrades to FAIL instead of aborting eval.

### Gaps (honest)

- End-to-end PASS improved only **+5.3 pp** on completed rows; MEDICAL/GENERAL regressed per-domain after prompt/sidecar experiments.
- Large surface area still **uncommitted** on `phase-1-repair-loops` (Docker, DSPy sidecars, many scripts) — portfolio should pin a release tag.
- `coding_optimized.txt` / most DSPy sidecars **not applied** to source (`apply_sidecar.py` exists; manual review gate).
- Free-tier **Groq TPD** drove ERROR rows and merge/resume complexity — not a production-grade dependency story.

### What would make it **9 / 10**

- All sidecars reviewed and applied (or rejected with A/B numbers).
- Single-command reproducible eval (`reproduce.sh merge` + pinned artifact checksums).
- Integration tests with mocked LLM/router for CI without secrets.

### Phase map

| Phase | Item | Status |
|---|---|---|
| **A** | CI workflow, judge/prose fixes, eval resume | **done** |
| **B** | `eval_naive_baseline.py`, `eval_repair_ablation.py`, RQ1 v2 wiring | **done** (v2 **blocked** on corpus) |
| **C** | `error_class`, `smoke_e2e.py`, ONNX export | **done** |
| **D** | `smoke_imports.py`, CI optional e2e job | **done** |
| **B** | Apply optimized prompts (`apply_sidecar.py`) | **pending** |
| **B** | Full 48-row naive baseline run | **pending** |

---

## 3. Eval rigor — **7 / 10** · **strong harness, modest scale**

### Evidence

- **Held-out set:** `data/eval_queries.jsonl` — 48 rows, stratified, tricky cases (HIPAA→LEGAL, prose coding→CODING); validated by `validate_eval_queries.py`.
- **Frozen router test:** 200 rows (`data/router/router_test.csv`); never used for RQ1 augment.
- **Baseline before repair** captured (`snapshot_baseline.py`); post-fix merged with explicit dedupe (`merge_experiment_rows`: prefer non-ERROR, newer wins).
- **FINAL merge** (`2026-07-11T12-36-52_post-fix-v2-merged`): **48/48** rows, **0 ERROR** (domain v1/v2 runs + `post-fix-resume-v1`).
- **Compare tooling:** `compare_experiments.py`, per-domain PASS in `snapshot_post_fix.py`, ERROR breakdown by `error_class` in eval summary.
- **Reproduce entry points:** `scripts/reproduce.sh`, `docs/RESULTS.md`, `docs/DEPLOY.md`.

### Gaps (honest)

- ***n* = 48** is fine for MVP, thin for statistical claims; no confidence intervals or multiple seeds.
- Post-fix FINAL is a **stitched merge** of 10 partial runs — methodologically sound for infra failures, but not a single clean `--execute` pass.
- **No automated full eval in CI** (by design — needs API keys; optional `smoke-e2e` is continue-on-error).
- **Human eval** / inter-rater judge calibration: none.

### What would make it **9 / 10**

- One fresh full eval pass with 0 ERROR without merge (or pre-registered merge plan in thesis).
- Bootstrap CIs on PASS% and McNemar on paired query outcomes (pre vs post).
- Held-out **second** query set for final thesis numbers (no prompt tuning on it).

### Phase map

| Phase | Item | Status |
|---|---|---|
| **A** | Eval harness, resume/merge, baseline snapshot | **done** |
| **B** | `docs/RESULTS.md`, naive vs pipeline comparison helpers | **done** |
| **C** | `smoke_e2e.py` (5 fixed queries) | **done** |
| **D** | `snapshot_post_fix.py` auto-merge + README FINAL table | **done** |
| **D** | Re-run eval after medical/legal prompt changes; refresh numbers | **pending** |

---

## 4. Research — **6 / 10** · **hypothesis clear, evidence mixed**

### Evidence

**End-to-end (primary product metric)**

- Completed PASS: **36.4% → 41.7%** (+5.3 pp). Meaningful but modest; committee should treat as directional.
- LEGAL/CODING repairs show targeted gains; MEDICAL/GENERAL do not.

**RQ1 (bootstrap corpus, synthetic/eval-export negatives)**

From `2026-07-11T08-38-52_rq1/manifest.json`:

| Arm | Router test acc. | Eval-queries routing |
|---|---:|---:|
| pre | 95.5% | **93.75%** |
| Run A (all negatives) | 99.0% | **97.92%** |
| Run B (ROUTER-only) | 99.0% | **97.92%** |
| **winner** | — | **tie** |

Conclusion in manifest: attribution-filtered retrain **does not separate** from blanket retrain at this corpus size (Run A augment *n* = 38, Run B *n* = 12).

**RQ1-bis (router-only ablation, fast)**

From `2026-07-11T11-22-52_repair_ablation/manifest.json`: arm0 **93.75%** → arm1/arm2 **97.92%** (+4.17 pp), **winner: tie**. Confirms repair helps routing; does not break Run A vs B tie.

**Orchestration ablation (naive vs ARCS)**

- `eval_naive_baseline.py` wired; same spec+judge as full pipeline.
- **Full 48-row naive run: not completed** (README still shows “run below”). Cannot yet quantify orchestration lift rigorously.

**RQ1 v2 (real demo feedback)**

- `feedback_corpus_real.jsonl` exists but **below thresholds** (≥40 negatives, ≥15 ROUTER-attributed). **Not run.**

### What would make it **9 / 10**

- RQ1 v2 on real feedback with **Run A ≠ Run B** (or powered negative result with adequate *n*).
- Completed **naive-baseline-v1** vs **post-fix-v2-merged** table with same completed-row denominator.
- End-to-end PASS **≥50%** on completed rows *or* clear per-domain story with ablation isolating each repair.

### Phase map

| Phase | Item | Status |
|---|---|---|
| **B** | RQ1 bootstrap pipeline + manifest | **done** (tie) |
| **B** | RQ1-bis router ablation | **done** (tie) |
| **B** | Naive baseline script + comparison | **done** (run **pending**) |
| **B** | RQ1 v2 real-feedback path | **done** (execution **pending**) |
| **B** | McNemar paired misroute analysis in `rq1_run.py` | **done** |
| **D** | Thesis-ready negative result write-up for RQ1 tie | **pending** |

---

## 5. Production readiness — **5 / 10** · **skeleton only**

### Evidence

- **Dockerfile** (multi-stage `.venv`), **docker-compose.yml**, **docs/DEPLOY.md**.
- **Health endpoint** `/health` — `groq_configured`, `nvidia_configured`, `router_backend`.
- **ONNX router** export + `smoke_router.py`; avoids PyTorch cold start in containers.
- **CI:** pytest gate (90 tests), pip cache, optional `smoke-e2e` (continue-on-error, needs secrets).
- **Logs contract:** `logs/README.md` documents `requests.jsonl`, `eval_failures.jsonl`, queues.

### Gaps (honest)

- **Not deployed** anywhere public; no TLS, auth, or multi-tenant isolation on demo API.
- Router weights **volume-mounted**, not baked — correct for dev, extra ops for prod.
- No metrics/tracing (Prometheus, structured log shipping), no alert on ERROR rate.
- Secrets via `.env` only; no rotation or secret manager story.

### What would make it **9 / 10**

- One stable deployed demo (Fly.io / Cloud Run / VM) with health + smoke in CD.
- ONNX default in prod, artifact versioning for router checkpoints.
- Rate-limit backoff as first-class UX (queue + user message), not exit code 2.

### Phase map

| Phase | Item | Status |
|---|---|---|
| **C** | Docker + compose + DEPLOY.md | **done** |
| **C** | ONNX router + `smoke_router.py` | **done** |
| **C** | `smoke_e2e.py` + optional CI job | **done** |
| **C** | Hosted deploy + monitoring | **pending** |
| **C** | Auth / rate-limit product layer | **pending** |

---

## 6. Documentation — **8 / 10** · **strong for a solo MVP**

### Evidence

- **README** (~1,100 lines): architecture diagram, Phase 1–3, eval workflow, RQ1/RQ1-bis, ONNX, Docker pointers, **Results (baseline vs post-fix)** with FINAL numbers.
- **docs/RESULTS.md** — paper-style (~4 pages): abstract, tables, limitations, reproduce commands.
- **docs/DEPLOY.md** — docker run, env vars, post-deploy verification.
- **logs/README.md** — log formats + `error_class` debugging.
- **scripts/reproduce.sh** — CI-safe check / eval-baseline / rq1-bootstrap / merge dry-runs.

### Gaps (honest)

- Much documentation references artifacts **not in git** (`artifacts/` gitignored) — reviewers must run pipelines or trust exported numbers.
- Some README footnotes cite older git SHA (`e60b30e8`); should refresh after release tag.
- No single **committee one-pager** PDF export (this SCORECARD is the start).

### What would make it **9 / 10**

- Release tag + artifact manifest (checksums for key `experiment.json` files).
- ARCHITECTURE.md with sequence diagrams for one query (happy path + repair path).
- SCORECARD refreshed automatically from `snapshot_post_fix.py` summary block.

### Phase map

| Phase | Item | Status |
|---|---|---|
| **B** | `docs/RESULTS.md` | **done** |
| **C** | `docs/DEPLOY.md` | **done** |
| **D** | README FINAL results table | **done** |
| **D** | `docs/SCORECARD.md` (this file) | **done** |
| **D** | Auto-refresh scorecard from artifacts | **pending** |

---

## Phase A–D checklist (prompt arc)

Development was organized in four agent phases. Use this for thesis “methods timeline.”

| Phase | Focus | Completed | Remaining |
|---|---|---|---|
| **A** | Reliability & CI — judge fixes, eval resume/merge, repair queues, GitHub Actions | CI, resume, merge, repair orchestrator, judge parse degrade | Pin release; mock-LLM CI tests |
| **B** | Research — RQ1 bootstrap, RQ1-bis, naive baseline, RESULTS.md, RQ1 v2 wiring | RQ1 manifest (tie), RQ1-bis (+4.17 pp routing), scripts + docs | RQ1 v2 real feedback, full naive eval, apply DSPy sidecars |
| **C** | Production path — Docker, ONNX, health, smoke tests, error taxonomy | Dockerfile, DEPLOY, ONNX, smoke_router/e2e, `error_class` | Hosted deploy, observability |
| **D** | Portfolio polish — FINAL merge, README numbers, reproduce, scorecard, CI smoke imports | `snapshot_post_fix` auto-merge, README table, SCORECARD, 90 tests | Post-prompt re-eval, tag + artifact bundle |

---

## Recommended committee narrative (one paragraph)

> ARCS demonstrates that **attribution-gated repair loops** can be engineered end-to-end: a router dispatches queries to verifiable domain pipelines, failures are blamed before retraining, and held-out eval shows **+5.3 percentage points** on completed PASS rate (36.4% → 41.7%) after targeted LEGAL/CODING repairs, with **zero ERROR rows** in the FINAL merged eval. Router retraining on bootstrap negative feedback improves held-out routing from **93.75% to 97.92%**, but **Run A vs Run B ties**, so attribution filtering remains **plausible but unconfirmed** until RQ1 v2 on real user feedback. The system is a **credible MVP** with strong documentation and eval hygiene; it is **not yet production-deployed**, and orchestration value vs a naive single-LLM baseline is **wired but not fully measured**.

---

## Next three actions (highest ROI for thesis)

1. **Run `eval_naive_baseline.py` on all 48 queries** — completes the orchestration claim ([Phase B](#phase-ad-checklist-prompt-arc)).
2. **Collect ≥40 real 👎 feedback rows** → RQ1 v2 — may finally separate Run A vs B ([Phase B](#phase-ad-checklist-prompt-arc)).
3. **Tag a release** + export `experiment.json` checksums for baseline, FINAL merge, and RQ1 manifest ([Phase D](#phase-ad-checklist-prompt-arc)).

---

*Last updated: 2026-07-11. Refresh after re-eval or RQ1 v2; link from [README](../README.md).*
