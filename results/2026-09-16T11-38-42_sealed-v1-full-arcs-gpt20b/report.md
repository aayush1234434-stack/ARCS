# Experiment report: sealed-v1-full-arcs-gpt20b

> Generated from the machine-readable experiment artifact. This report does not
> replace human review of individual rows.

## Outcome

| Completed | PASS | FAIL | ERROR | PASS rate | 95% Wilson CI |
|---:|---:|---:|---:|---:|---:|
| 21 | 6 | 15 | 1 | 28.6% | 13.8%–50.0% |

## Per-domain results

| Domain | n | PASS rate |
|---|---:|---:|
| CODING | 10 | 40.0% |
| MEDICAL | 10 | 20.0% |
| LEGAL | 2 | 0.0% |
| GENERAL | 0 | n/a |

## Provenance

- Run ID: `2026-09-16T11-38-42_sealed-v1-full-arcs-gpt20b`
- Created: `2026-09-16T11:38:42.334663+00:00`
- Git commit: `c686854db1f57def598ce355a7e331a3676d44e0`
- Dirty working tree: `False`
- Schema: `1.0`
- Python: `3.11.5`
- Source artifact SHA-256: `6d277ea23e1c8e544e00f61c371cf1ffc1e89320af081d34d54473a8bb6f7c42`
- Mean / p95 latency: `45797.591 / 105185.0 ms`
- Measured API calls / tokens: `75 / 109376`

### Dataset fingerprints

| Path | Bytes | SHA-256 |
|---|---:|---|
| `data/eval_queries.jsonl` | 9056 | `75b8b14778beb19e9d65897267a178e6ea446508f064c92d64bf94353b73b89f` |
| `data/router/router_train.csv` | 46064 | `b9b165833e9f114d8476330aa7c81a932bb00a603b3bf49ff88bdeba3b400b31` |
| `data/router/router_test.csv` | 11346 | `f4a2e80dd5fbcb2f04be2a6e566e69742a89bebf256ff716accb8642e0aa0c1d` |
| `data/sealed/arcs_sealed_pilot_v1.jsonl` | 14522 | `a66c29c07a601f559c205ff5950d3b51789960a2922c5a77230bc38a397ae8f3` |

### Non-secret configuration

```json
{
  "coding_max_retries": 3,
  "generator_model": "openai/gpt-oss-20b",
  "judge_model": "meta/llama-3.2-11b-vision-instruct",
  "judge_strict": "1",
  "router_backend": "sklearn",
  "router_confidence_threshold": 0.75,
  "spec_model": "qwen/qwen3.8-27b",
  "test_generator_model": "qwen/qwen3.8-27b"
}
```

## Interpretation limits

- LLM-judge PASS is a proxy metric, not a substitute for blinded human evaluation.
- Report repeated runs when model sampling or provider behavior can vary.
- Do not tune prompts or thresholds against a sealed test split.
- Inspect and publish redacted row-level evidence before making comparative claims.
