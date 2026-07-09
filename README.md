# ARCS — Adaptive Routing & Correction System

> A modular orchestration architecture that routes each query to a domain-specific processing pipeline, verifies the answer before delivery, and learns from attributed failures.

**Status: In active development **

---

## What ARCS Is

ARCS is **not** “four separate specialist brains pretending via prompts.” It is an orchestration system built around one idea most systems skip: **attribution before retraining**.

A **specialist pipeline** is defined by:

- prompting strategy
- structured output contract
- verification mechanism
- optional toolchain (e.g. sandbox retries)

The underlying language model is **interchangeable**. The same orchestration runs with one general-purpose generator today, or a heterogeneous pool of domain models later, without rewriting the pipeline.

When a user signals that an answer was wrong, ARCS assigns blame to a component (router, verifier, specialist/pipeline, or ambiguous) before any retraining occurs.

---

## How It Works

```
User query
    │
    ▼
┌─────────────────────────────────────┐
│  Router — DistilBERT classifier     │
│  domain label + confidence score    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Resolve domain Pipeline            │
│  prompt · contract · verifier · tools│
│  (model from config / env override) │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Spec Generator (separate family)   │
│  builds expected answer checklist   │
└──────────────────┬──────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │ CODING                      │ MEDICAL / LEGAL / GENERAL
    ▼                             ▼
Sandbox                       Judge LLM
(independent tests ·           (spec coverage · confidence)
 retry ×3)
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼
          Answer delivered to user
                   │
                   ▼  (post-inference)
┌─────────────────────────────────────┐
│  Feedback + Attribution Engine      │
│  ROUTER · VERIFIER · SPECIALIST ·   │
│  AMBIGUOUS (discarded, not trained) │
└─────────────────────────────────────┘
```

---

## Specialist Pipelines (current MVP)

| Pipeline | Verifier | Toolchain | Default generator |
|---|---|---|---|
| `CODING` | Docker / subprocess sandbox | Independent test generator + up to 3 retries | `llama-3.3-70b-versatile` (Groq) |
| `MEDICAL` | LLM judge | Spec checklist | same (env-overridable) |
| `LEGAL` | LLM judge | Spec checklist | same (env-overridable) |
| `GENERAL` | LLM judge | Spec checklist | same (fallback / low confidence) |

Cross-checks deliberately use **different model families** where possible:

| Role | Default |
|---|---|
| Domain generator | Llama 3.3 70B (Groq) |
| Spec generator | Qwen3 32B (Groq) |
| Coding test generator | Qwen3 32B (Groq) |
| Judge | Llama 3.1 8B Instruct (NVIDIA) |

Override models without code changes:

```bash
export ARCS_GENERATOR_MODEL=llama-3.3-70b-versatile
export ARCS_CODING_MODEL=...      # optional domain override
export ARCS_SPEC_MODEL=qwen/qwen3-32b
export ARCS_TEST_GENERATOR_MODEL=qwen/qwen3-32b
export NVIDIA_JUDGE_MODEL=meta/llama-3.1-8b-instruct
export ARCS_CODING_MAX_RETRIES=3
```

When you later host CodeLlama / BioMistral (or other domain models), set `ARCS_CODING_MODEL` / `ARCS_MEDICAL_MODEL` — orchestration stays the same.

---

## Components

### 1. Router
Fine-tuned DistilBERT (ONNX by default). Outputs domain + confidence + full score distribution. Below 0.75 confidence → `GENERAL` pipeline.

### 2. Pipeline registry (`arcs/pipelines/registry.py`)
Maps each domain to its specialist module, verifier kind, retries, and tools. This is the modular contract; specialists are not hard-wired into `main.py` as “magic models.”

### 3. Spec Generator
Builds a checklist of required / correctness / disqualifying criteria before verification.

### 4. Verification

**Executable (`CODING`)** → independent test snippets + sandbox. Up to N rounds with failure feedback back into the coding specialist.

**Non-executable** → Judge LLM against the spec (pass only if score ≥ 0.75 and no missing required / disqualifying items).

### 5. Attribution Engine
Rule-based blame assignment from the evidence trail (sandbox fail → specialist; high-confidence verifier + negative feedback → verifier; low router confidence → router; etc.). `AMBIGUOUS` is discarded — not trained on.

---

## Project layout

```
ARCS/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                        # CLI shim → arcs.main
│
├── arcs/                          # main Python package
│   ├── main.py                    # orchestrator
│   ├── config.py                  # paths + model defaults
│   ├── progress.py
│   ├── router/                    # DistilBERT classifier
│   ├── pipelines/                 # domain registry + specialists
│   ├── verification/              # judge, sandbox, spec/test generators
│   ├── post/                      # feedback, attribution, logger
│   └── clients/                   # Groq client
│
├── data/router/                   # router training CSVs
├── artifacts/                     # router-model, checkpoints, eval-results
├── logs/                          # runtime JSONL (gitignored)
└── scripts/                       # one-off utilities
```

---

## Run

```bash
cd ARCS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY and NVIDIA_API_KEY

python main.py "Write a Python function that reverses a string."
python main.py --feedback NEGATIVE "What is the max safe dose of acetaminophen?"
python main.py --no-feedback --quiet "..."
```

Equivalent: `python -m arcs.main "..."`

### Demo web UI (presentations)

Clean ask + 👍/👎 feedback for demos (no CLI flags):

```bash
# From the project root (folder that contains main.py, scripts/, arcs/)
cd /Users/aayushsingh/Developer/ARCS

# Use the project venv — NOT conda (base) pip
source .venv/bin/activate

# Only if fastapi is missing:
pip install fastapi uvicorn

python scripts/run_demo.py
# Open http://127.0.0.1:8000
```

**Common mistakes:**
- Do not run `cd ARCS` twice — if your prompt already ends in `ARCS`, stay there.
- Do not run from inside `arcs/` — `scripts/run_demo.py` lives at the **repo root**.
- If `source .venv/bin/activate` fails, create the venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

- **Ask** runs the full pipeline and logs to `logs/requests.jsonl`
- **👍 / 👎** records feedback + attribution (same as CLI `--feedback`)
- After collecting NEGATIVE feedback: `python scripts/extract_queues.py`

Router training / eval (from project root):

```bash
python -m arcs.router.train
python -m arcs.router.evaluate
python scripts/export_router_onnx.py
```

Requires `GROQ_API_KEY` and `NVIDIA_API_KEY` in `.env`. Docker is preferred for sandbox isolation; a restricted local subprocess fallback is used if Docker is unavailable.

---

## Batch data collection

Run many queries (with optional feedback labels) to grow `logs/requests.jsonl` for retraining experiments:

```bash
# Preview what would run (no API calls)
python scripts/run_batch.py --dry-run

# Run first 3 queries only
python scripts/run_batch.py --limit 3 --quiet

# Full batch (uses feedback fields from the file when present)
python scripts/run_batch.py --quiet

# Ignore feedback columns in the batch file
python scripts/run_batch.py --no-feedback --quiet
```

Seed file: `data/batch_queries.jsonl` — one JSON object per line:

| Field | Required | Description |
|---|---|---|
| `query` | yes | User question |
| `feedback` | no | `POSITIVE`, `NEGATIVE`, or `null` |
| `expected_domain` | no | `CODING` / `MEDICAL` / `LEGAL` / `GENERAL` (for later router labels) |

This is **data collection only** — it does not retrain the router or run DSPy. After collecting NEGATIVE feedback, run `python scripts/extract_queues.py`.

---

## Retraining queues

After collecting NEGATIVE feedback, sort failures into per-component piles for repair:

```bash
python scripts/extract_queues.py
python scripts/extract_queues.py --dry-run          # counts only
python scripts/extract_queues.py --input logs/requests.jsonl --output-dir logs/queues
```

This reads `logs/requests.jsonl`, keeps rows with `user_feedback == NEGATIVE`, and buckets by `attribution.component` into:

| File | Use for |
|---|---|
| `logs/queues/router_queue.jsonl` | Router DistilBERT retrain |
| `logs/queues/specialist_queue.jsonl` | Specialist prompt / DSPy |
| `logs/queues/verifier_queue.jsonl` | Judge prompt / DSPy |
| `logs/queues/ambiguous_queue.jsonl` | Discard / review (do not train) |

Filter and groupby only — attribution already decided blame.

---

## Router repair path

Close the DistilBERT retrain loop for ROUTER-attributed failures:

```bash
# 1. Sort NEGATIVE logs into per-component queues
python scripts/extract_queues.py

# 2. Label router failures with the *correct* domain
#    (do not use route.domain — that may be the wrong prediction)
python scripts/label_router_failures.py --list
python scripts/label_router_failures.py --query-id <uuid> --correct-domain MEDICAL
# or interactively:
python scripts/label_router_failures.py --interactive

# 3. Export labeled rows → data/router/router_train.csv and retrain
python scripts/retrain_router.py
# export only (no training):
python scripts/retrain_router.py --skip-train
# assume queues already exist:
python scripts/retrain_router.py --skip-extract
```

Flow: `extract_queues` → `label_router_failures` → `retrain_router`.

- Labels come from `expected_domain` (batch file) or `correct_domain` (manual).
- Unlabeled router-queue rows are skipped with a warning.
- Duplicate `(text, label)` pairs are not appended again.
- This path does **not** use DSPy — DistilBERT retrain only.

---

## MEDICAL prompt optimization (DSPy)

Use SPECIALIST-attributed MEDICAL failures to propose a better system prompt.
DSPy writes a **sidecar file** — it never overwrites `medical.py` automatically.

```bash
pip install -r requirements.txt   # includes dspy==3.2.1

# 1. Collect NEGATIVE feedback + extract queues
python scripts/run_batch.py --quiet
python scripts/extract_queues.py

# 2. Preview which MEDICAL examples would be used
python scripts/optimize_medical.py --dry-run --max-examples 20

# 3. Run COPRO (calls Groq + NVIDIA judge; can take a while)
python scripts/optimize_medical.py --max-examples 20
```

Output: `artifacts/prompts/medical_optimized.txt`

### Review and apply manually

1. Open `artifacts/prompts/medical_optimized.txt`
2. Compare against `SYSTEM_PROMPT` in `arcs/pipelines/specialists/medical.py`
3. If the new prompt looks better (clearer structure, safer caveats, better claim format), **copy it by hand** into `SYSTEM_PROMPT`
4. Re-run a few medical queries and check judge scores before committing

Metric: existing LLM judge (`arcs.verification.judge`) — PASS / score. Sandbox is N/A for MEDICAL.

---

## Verifier optimization (DSPy)

When attribution blames **VERIFIER** (user said NEGATIVE but the judge passed / scored high), optimize the LLM judge system prompt. Sandbox path is out of scope.

```bash
python scripts/extract_queues.py

# Preview false-PASS examples from verifier_queue.jsonl
python scripts/optimize_judge.py --dry-run --max-examples 20

# Run COPRO (NVIDIA judge LM; can take a while)
python scripts/optimize_judge.py --max-examples 20
```

Output: `artifacts/prompts/judge_optimized.txt`

### Review and apply manually

1. Open `artifacts/prompts/judge_optimized.txt`
2. Compare against `SYSTEM_PROMPT` in `arcs/verification/judge.py`
3. If the new prompt is stricter on known-bad answers (more FAIL / lower scores), **copy it by hand** into `SYSTEM_PROMPT`
4. Re-check a few previously false-PASS cases before committing

Metric: on known-bad answers (`expected_verdict=FAIL`), the optimized judge should return **FAIL** or score &lt; 0.75. Parsing reuses `judge._extract_json` / `_normalize_result`.

---

## Research Questions

### RQ1 — Attribution-filtered retraining *(primary)*

> Does training the router only on failures attributed to the router produce a more accurate classifier than training on all negative feedback?

**Run A:** retrain on every negative signal.  
**Run B:** retrain only where `attribution == ROUTER`.

### RQ2 — Heterogeneous specialists vs one large generalist *(future)*

Only meaningful once domain-specific models (or clearly stronger domain backends) are plugged into the same pipeline slots. Prompt-only cosplay does **not** answer RQ2.

---

## Tech stack (as implemented)

| Component | Technology |
|---|---|
| Router | DistilBERT + ONNX Runtime |
| Pipelines | `arcs/pipelines/` + domain prompt/contracts |
| Generator (default) | Groq `llama-3.3-70b-versatile` |
| Spec / tests | Groq `qwen/qwen3-32b` |
| Judge | NVIDIA OpenAI-compatible API |
| Sandbox | Docker (`python:3.11-slim`) with subprocess fallback |
| Logging | JSONL (`logs/requests.jsonl`) |

Planned / not in this MVP: ChromaDB grounding, DSPy batch prompt optimization, multi-domain query decomposition, autonomous evaluator feedback.

---

## Known limitations

- With one shared generator, domain value comes from **pipeline structure** (contracts + verification + tools), not from invented specialist weights.
- Attribution is heuristic; `AMBIGUOUS` reduces noisy training.
- Feedback is currently explicit / interactive only.
- Verifier miscalibration remains the most dangerous failure mode — recalibrate against human labels periodically.

---

*Questions or ideas: aayush1234434@gmail.com*
