# ARCS — Committee / Portfolio Presentation

*10-minute talking outline. Numbers from `2026-07-11T12-36-52_post-fix-v2-merged` and RQ1 manifest.*

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
| Post-fix FINAL | 20 | 28 | 0 | **41.7%** |

**+5.3 pp** on completed rows; **0 ERROR** after merge/resume.

Per-domain highlights:

| Domain | Δ |
|---|---:|
| LEGAL | **+31 pp** |
| CODING | **+17 pp** |
| MEDICAL | −8 pp |
| GENERAL | −9 pp |

Repairs were targeted; regressions show the cost of prompt experiments without full re-eval.

---

## 4. Results — RQ1 (2 min)

**Hypothesis:** Router retrain on ROUTER-attributed negatives (Run B) beats blanket negatives (Run A).

| Arm | Eval routing | Router test |
|---|---:|---:|
| pre | 93.75% | 95.5% |
| Run A (all negatives) | 97.92% | 99.0% |
| Run B (ROUTER-only) | 97.92% | 99.0% |

**Winner: tie.** Retraining helps (+4.17 pp routing); attribution filtering **not distinguished** on bootstrap corpus (*n* augment: 38 vs 12).

**Next:** RQ1 v2 on real demo feedback (≥40 negatives, ≥15 ROUTER).

---

## 5. Engineering credibility (1 min)

- **90 pytest tests**, CI on push, optional smoke-e2e with secrets
- Eval **resume/merge** after Groq TPD; `error_class` for debugging
- **Docker + ONNX router** path for production-shaped deploy
- Full reproducibility docs: [RESULTS.md](RESULTS.md), [reproduce.sh](../scripts/reproduce.sh)

---

## 6. Honest limitations (1 min)

- Small eval set (*n* = 48); post-fix FINAL is a **stitched merge** of partial runs
- **Naive baseline** (single LLM, same judge) wired but not run at full scale
- **Not deployed** publicly; free-tier API limits drove infra complexity
- MEDICAL/GENERAL repairs did not generalize in this cycle

See [SCORECARD.md](SCORECARD.md) for 1–10 self-ratings.

---

## 7. Ask / next steps (1 min)

1. Run naive baseline → quantify orchestration lift
2. Collect real user 👎 → RQ1 v2
3. Tag release + artifact checksums for thesis appendix

---

## Slide checklist

- [ ] Architecture diagram (README)
- [ ] Baseline vs post-fix table
- [ ] RQ1 tie chart (pre / Run A / Run B)
- [ ] Demo screenshot or 30s screen recording
- [ ] SCORECARD summary radar or table
- [ ] Limitations slide (required for committee trust)
