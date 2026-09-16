# ARCS — Adaptive Routing & Correction System

> Most LLM apps answer every question with one prompt and hope it's right. ARCS routes each query to a domain pipeline, **verifies the answer before delivery**, and, when a user says it's wrong, assigns blame to a component *before* any retraining.

[![CI](https://github.com/aayush1234434-stack/ARCS/actions/workflows/ci.yml/badge.svg)](https://github.com/aayush1234434-stack/ARCS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-3d9b7a.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab.svg)](pyproject.toml)

**One-liner architecture:** `route (deterministic sklearn; optional torch/ONNX) → resolve domain pipeline → generate answer → build spec → verify (sandbox for code, LLM judge for prose) → deliver → attribute feedback`.

---

## Sealed pilot status

ARCS now has a frozen 40-query pilot benchmark with 10 queries each for
CODING, MEDICAL, LEGAL, and GENERAL. It was hashed before the first model run:
`a66c29c07a601f559c205ff5950d3b51789960a2922c5a77230bc38a397ae8f3`.

| Condition | Rows observed | PASS | FAIL | ERROR | PASS rate over completed rows |
|---|---:|---:|---:|---:|---:|
| Naive single-LLM baseline | 40/40 | 13 | 27 | 0 | 32.5% |
| ARCS without runtime verification | 40/40 | 11 | 29 | 0 | 27.5% |
| Full ARCS | 22/40 | 6 | 15 | 1 | 28.6% |

All three matched pilot conditions used `openai/gpt-oss-20b` as the generator,
Qwen 3.8 27B for specifications/tests, and NVIDIA Llama 3.2 11B as the judge.
The full-ARCS condition is **incomplete**: Groq's 200,000-token daily quota
stopped the run after 21 completed rows plus one quota error. Its 28.6% figure
must not be presented as a result for the complete 40-query set, and the pilot
does not currently establish that ARCS outperforms the naive baseline.

Redacted row-level artifacts, checksums, provenance, and the exact limitation
are published in [`results/`](results/README.md). A blinded pilot packet has
also been prepared for two independent reviewers using the seven queries for
which all three systems produced answers; human ratings have not yet been
collected.

---

## The development-benchmark story

Development set: **48 multi-domain queries** (`data/eval_queries.jsonl`), PASS% over completed rows (ERROR rows excluded from the denominator).

| Milestone | PASS | What changed |
|---|---:|---|
| Baseline v1 | **36.4%** | First end-to-end run (16/44) |
| Post-fix v2 | **47.9%** | Prompt hardening + coding-path fixes + router retrain (23/48) |
| Naive single-LLM | **62.5%** | Same generator + judge, **no** orchestration (30/48) |
| **ARCS after naive-gap fix** | **66.7%** | Coding judge-fallback + specialist completeness (32/48) |

The naive baseline was an uncomfortable result worth keeping honest: for a while a single LLM call **beat** orchestrated ARCS (62.5% vs 47.9%, −14.6 pts). Diagnosing that gap — CODING sandbox mismatches and incomplete MEDICAL/GENERAL answers — is what pushed ARCS to **66.7%**, now **+4.2 pts** ahead of naive. Full tables, per-domain breakdowns, and the historical −14.6 pt finding live in [docs/RESULTS.md](docs/RESULTS.md).

> **Evidence status:** these are historical development-set results, not a sealed-test
> generalization claim. Prompt and pipeline fixes were informed by failures on this set.
> Local source artifacts were not committed in the original work, so a reviewer cannot
> independently audit every row from this clone. New runs capture dataset hashes,
> configuration, git state, confidence intervals, latency, and token usage; publish them
> with `scripts/publish_experiment.py`. See [the benchmarking protocol](docs/BENCHMARKING.md).

**RQ1 (attribution-filtered retraining): a tie, reported honestly.** On a bootstrap corpus (38 synthetic negatives), retraining the router lifted eval-query routing accuracy **93.75% → 97.92%** — but Run A (all negatives) and Run B (ROUTER-only) landed on **identical** metrics. Attribution filtering is **inconclusive at this sample size, not refuted**. RQ1 v2 on real 👍/👎 feedback (needs ≥40 negatives, ≥15 ROUTER-attributed) is scoped as future work. Details in [docs/RESULTS.md](docs/RESULTS.md).

---

## Why this is an *orchestration* system, not four prompts

A **specialist pipeline** is defined by four things, not by a magic model:

- prompting strategy
- structured output contract
- verification mechanism
- optional toolchain (e.g. sandbox retries)

The underlying LLM is **interchangeable** — one general generator today, a heterogeneous pool of domain models later, with no pipeline rewrite. The idea most systems skip is **attribution before retraining**: when feedback is negative, ARCS blames the router, verifier, specialist, or marks it `AMBIGUOUS` (discarded, never trained on) *before* touching any weights.

```
User query
   │
   ▼
Router (sklearn default; optional torch/ONNX) ── domain + confidence (< 0.75 → GENERAL)
   │
   ▼
Resolve domain Pipeline ── prompt · contract · verifier · tools
   │
   ▼
Spec Generator (separate model family) ── expected-answer checklist
   │
   ├── CODING ────────────► Sandbox (independent tests, retry ×3)
   │                          └─ still failing but non-empty? → LLM judge
   └── MEDICAL/LEGAL/GENERAL ► LLM judge (spec coverage, score ≥ 0.75)
   │
   ▼
Answer delivered
   │
   ▼ (post-inference)
Feedback + Attribution ── ROUTER · VERIFIER · SPECIALIST · AMBIGUOUS
```

Default model families are deliberately mixed so verification is a real
cross-check: generator **GPT-OSS 120B** (Groq), spec + coding tests
**Qwen 3.8 27B** (Groq), and judge **Llama 3.2 11B Vision Instruct** (NVIDIA).
All are overridable by env (`ARCS_GENERATOR_MODEL`, `ARCS_SPEC_MODEL`,
`ARCS_TEST_GENERATOR_MODEL`, `NVIDIA_JUDGE_MODEL`, …) with no code changes.

---

## Quick start

```bash
cd ARCS
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,optimization]"
cp .env.example .env          # add GROQ_API_KEY and NVIDIA_API_KEY

# Ask a question through the full pipeline
python main.py "Write a Python function that reverses a string."
python main.py --feedback NEGATIVE "What is the max safe dose of acetaminophen?"
```

Demo web UI (ask + 👍/👎 feedback, good for presentations):

```bash
source .venv/bin/activate
python scripts/run_demo.py     # open http://127.0.0.1:8000
```

No keys yet? Tour the routing and trace interface with deterministic, explicitly
labeled sample output:

```bash
ARCS_DEMO_OFFLINE=1 python scripts/run_demo.py
```

The response UI exposes routing confidence, specification, model, verifier,
rounds, timing, and measured token usage. Offline output is never valid benchmark
evidence.

Docker is required for generated-code execution. The sandbox fails closed when
secure container isolation is unavailable; host execution is disabled by default.
See [SECURITY.md](SECURITY.md) and [deployment guidance](docs/DEPLOY.md).

---

## Evaluation you can audit

Every new experiment records the git commit and dirty state, Python/platform
details, non-secret model configuration, dataset SHA-256 hashes, latency, and a
95% Wilson confidence interval.

```bash
# Plan a run without API calls
python scripts/eval_pipeline.py --dry-run

# Run and save locally
python scripts/eval_pipeline.py --name sealed-test-v1

# Create a redacted, checksummed bundle suitable for review/commit
python scripts/publish_experiment.py artifacts/experiments/<run-id>

# Prepare a blinded packet for two or more human reviewers
python scripts/prepare_human_eval.py \
  --system naive=artifacts/experiments/<naive-run> \
  --system arcs=artifacts/experiments/<arcs-run>
```

The required development/validation/sealed-test split and human-review protocol
are defined in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

---

## Project layout

```
ARCS/
├── main.py                 # CLI shim → arcs.main
├── arcs/
│   ├── main.py             # orchestrator
│   ├── router/             # deterministic sklearn default; optional torch / ONNX
│   ├── pipelines/          # domain registry + specialists
│   ├── verification/       # judge, sandbox, spec/test generators
│   ├── post/               # feedback, attribution, logger
│   └── clients/            # Groq / NVIDIA clients
├── data/                   # eval_queries.jsonl, router CSVs, batch seeds
├── scripts/                # eval, repair, DSPy optimize, demo, router tools
├── artifacts/experiments/  # saved eval runs (gitignored except README)
├── results/                # curated, redacted, checksummed evidence bundles
└── docs/                   # RESULTS, DEPLOY, PRESENTATION, SCORECARD
```

---

## Documentation

| Doc | What's inside |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Paper-style tables, naive vs ARCS, RQ1 / RQ1-bis, judge ablation, full reproduce commands |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Docker, env vars, ONNX router deployment, health check |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Split discipline, repeated runs, human review, evidence publishing |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | 10-minute committee / portfolio outline |
| [docs/SCORECARD.md](docs/SCORECARD.md) | Honest 1–10 self-ratings |

The full operator playbook — evaluation harness, resume/merge after Groq quota limits, Phase 1–3 repair loops, DSPy prompt optimization, and RQ1 reproduce steps — lives in [docs/RESULTS.md](docs/RESULTS.md) so this README stays a quick tour.

---

## Known limitations

- With one shared generator, domain value comes from **pipeline structure** (contracts + verification + tools), not invented specialist weights — RQ2 (heterogeneous specialists) is future work.
- Attribution is heuristic; `AMBIGUOUS` rows are discarded to reduce noisy training.
- Feedback is currently explicit/interactive only; RQ1 v2 needs real accumulated 👎 signal.
- Verifier miscalibration is the most dangerous failure mode — recalibrate against human labels periodically.
- Single-run eval at *n* = 48 is sensitive to judge variance; treat point PASS rates as directional.
- The sealed pilot has only 40 queries, its full-ARCS condition is incomplete,
  and its blinded packet is not yet rated; a larger completed sealed test and
  two-reviewer study are still required for a general performance claim.

---

*Questions or ideas: aayush1234434@gmail.com*
