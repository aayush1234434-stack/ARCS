# ARCS — Adaptive Routing & Correction System

> A modular orchestration architecture that routes each query to a domain-specific processing pipeline, verifies the answer before delivery, and learns from attributed failures.

**Status: In active development**

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

### 2. Pipeline registry (`pipelines.py`)
Maps each domain to its specialist module, verifier kind, retries, and tools. This is the modular contract; specialists are not hard-wired into `main.py` as “magic models.”

### 3. Spec Generator
Builds a checklist of required / correctness / disqualifying criteria before verification.

### 4. Verification

**Executable (`CODING`)** → independent test snippets + sandbox. Up to N rounds with failure feedback back into the coding specialist.

**Non-executable** → Judge LLM against the spec (pass only if score ≥ 0.75 and no missing required / disqualifying items).

### 5. Attribution Engine
Rule-based blame assignment from the evidence trail (sandbox fail → specialist; high-confidence verifier + negative feedback → verifier; low router confidence → router; etc.). `AMBIGUOUS` is discarded — not trained on.

---

## Run

```bash
cd ARCS
python main.py "Write a Python function that reverses a string."
python main.py --feedback NEGATIVE "What is the max safe dose of acetaminophen?"
python main.py --no-feedback --quiet "..."
```

Requires `GROQ_API_KEY` and `NVIDIA_API_KEY` in `.env`. Docker is preferred for sandbox isolation; a restricted local subprocess fallback is used if Docker is unavailable.

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
| Pipelines | `pipelines.py` + domain prompt/contracts |
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
