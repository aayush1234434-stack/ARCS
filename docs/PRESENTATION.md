# ARCS — Committee / Portfolio Presentation

*10-minute talking outline. Numbers from `2026-07-11T13-45-31_post-fix-v2-merged` and RQ1 bootstrap manifest.*

---

## 1. Problem (1 min)

Single-model chatbots mix routing, generation, and verification in one undifferentiated call. When something fails, you cannot tell whether to fix the router, the specialist prompt, or the verifier — so “retrain everything” wastes effort and can hurt unrelated domains.

**Claim:** Attribution-gated repair loops need an orchestration architecture, not better prompts alone.

---

## 2. System (2 min)

**ARCS** = route → specialist pipeline → spec → verify → attributed feedback → targeted repair.

```
Query → DistilBERT router → domain pipeline → spec (Qwen) → sandbox/judge → answer
                                                      ↓
                                            feedback + blame (ROUTER / SPECIALIST / VERIFIER)
```

- **48-query held-out eval** across CODING, MEDICAL, LEGAL, GENERAL
- **Independent verifier** per domain (sandbox for code, LLM judge for prose)
- **Repair queues** feed router retrain or DSPy prompt optimization — not blanket fine-tuning

Demo: `python -m arcs.demo.app` or Docker ([DEPLOY.md](DEPLOY.md)).

---

## 3. Results — end-to-end (2 min)

| Run | PASS | FAIL | ERROR | PASS% (completed) |
|---|---:|---:|---:|---:|
| Baseline | 16 | 28 | 4 | **36.4%** |
| Post-fix FINAL | 23 | 25 | 0 | **47.9%** |

**+11.5 pp** on completed rows; **0 ERROR** in canonical FINAL merge (48/48).

Per-domain highlights:

| Domain | Δ |
|---|---:|
| CODING | **+42 pp** |
| LEGAL | **+31 pp** |
| MEDICAL | −8 pp |
| GENERAL | −9 pp |

Repairs were targeted; MEDICAL/GENERAL did not gain in this cycle.

---

## 3b. Specialist repair loop — CODING (1 min)

**Attribution → queue → DSPy sidecar → human review → apply → re-eval**

```
eval FAIL (SPECIALIST) → specialist_queue.jsonl
  → optimize_coding.py → artifacts/prompts/coding_optimized.txt
  → apply_sidecar.py (dry-run diff, then .bak + apply)
  → eval_pipeline --domains CODING
```

| Step | Tool | Notes |
|---|---|---|
| Extract | `repair.py --component SPECIALIST --domain CODING` | From demo/eval failures |
| Optimize | `optimize_coding.py` | COPRO on queue; **needs GROQ**; writes sidecar only |
| Apply | `apply_sidecar.py --dry-run` then apply | Never auto-applied; creates `coding.py.bak` |
| Verify | `eval_pipeline --domains CODING --name post-fix-coding-v4` | CODING PASS: **25% → 67%** in FINAL merge |

One-shot: `./scripts/run_coding_repair.sh` (use `--dry-run` to plan without API writes). Groq COPRO defaults: **`--breadth 2 --depth 2`** (`n=1` per API call).

**Status:** `coding_optimized.txt` not in repo — generate locally; see [artifacts/prompts/README.md](../artifacts/prompts/README.md).

---

## 4. Results — RQ1 bootstrap (2 min)

**Hypothesis:** Router retrain on ROUTER-attributed negatives (Run B) beats blanket negatives (Run A).

| Arm | Eval routing | Router test |
|---|---:|---:|
| pre | 93.75% | 95.5% |
| Run A (all negatives) | 97.92% | 99.0% |
| Run B (ROUTER-only) | 97.92% | 99.0% |

**Outcome: tie (bootstrap complete).** Retraining helps (+4.17 pp routing); attribution filtering **inconclusive** at bootstrap *N* (38 vs 12 augment rows). Corpus is synthetic — reconstructed from eval artifacts, not live 👎.

**RQ1 v2 (real feedback):** future work — requires ≥40 👎 and ≥15 ROUTER-attributed (currently **2 / 0**). Bootstrap manifest is the authoritative RQ1 result until v2 runs.

---

## 5. Engineering credibility (1 min)

- **119 pytest tests**, CI on push, optional smoke-e2e with secrets
- Eval **resume/merge** after Groq TPD; `error_class` for debugging
- **Docker + ONNX router** path for production-shaped deploy
- Full reproducibility docs: [RESULTS.md](RESULTS.md), [reproduce.sh](../scripts/reproduce.sh)

---

## 6. Honest limitations (1 min)

- Small eval set (*n* = 48); post-fix FINAL is a **stitched merge** of partial runs
- **RQ1 bootstrap tied** — attribution hypothesis not confirmed at this *N*; v2 explicitly deferred
- **Naive baseline** (single LLM, same judge) wired but not run at full scale
- **Not deployed** publicly; free-tier API limits drove infra complexity

See [SCORECARD.md](SCORECARD.md) for 1–10 self-ratings.

---

## 7. Scope closed / optional follow-ups (1 min)

**Shipped in MVP (this document):**

1. End-to-end orchestration + eval harness + **47.9%** FINAL merge (48/48, 0 ERROR)
2. Bootstrap RQ1 controlled A/B — **tie reported honestly**
3. Reproduce path + tagged release (`v1.0.0-mvp`)

**Explicitly future work (not blockers):**

1. RQ1 v2 on real demo feedback (≥40 / ≥15 gates)
2. Full 48-row naive baseline ablation
3. Path from 47.9% → 60% PASS (specialist + judge levers in [RESULTS.md §7](RESULTS.md#7-path-to-60--specialist--judge-levers-2026-07-11))

---

## Slide checklist

- [ ] CODING specialist repair flow (section 3b)
- [ ] Baseline vs post-fix table (47.9%)
- [ ] RQ1 tie chart (pre / Run A / Run B) + “bootstrap complete, v2 future”
- [ ] Demo screenshot or 30s screen recording
- [ ] SCORECARD summary radar or table
- [ ] Limitations slide (required for committee trust)
