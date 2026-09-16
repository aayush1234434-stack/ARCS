# ARCS data card

## Intended use

| File | Rows | Intended role | SHA-256 |
|---|---:|---|---|
| `eval_queries.jsonl` | 48 | Development benchmark only | `75b8b14778beb19e9d65897267a178e6ea446508f064c92d64bf94353b73b89f` |
| `router/router_train.csv` | 800 | Router training | `b9b165833e9f114d8476330aa7c81a932bb00a603b3bf49ff88bdeba3b400b31` |
| `router/router_test.csv` | 200 | Frozen router-only test | `f4a2e80dd5fbcb2f04be2a6e566e69742a89bebf256ff716accb8642e0aa0c1d` |

The 48-query file is not a sealed test set. Its failures have already informed
prompt, routing, and verification changes. It is suitable for debugging and
regression checks, not for a final generalization claim.

## Composition

The datasets cover four labels: `CODING`, `MEDICAL`, `LEGAL`, and `GENERAL`.
The development benchmark includes both straightforward and intentionally
ambiguous or cross-domain prompts. It is too small to represent real traffic,
rare safety failures, jurisdictional diversity, or demographic variation.

## Provenance and licensing limitation

The original repository does not contain a complete prompt-by-prompt provenance
record or annotator log. Until that is reconstructed, treat the data as
author-curated project data and do not redistribute it as a general public
benchmark. New datasets should record source, license, author, creation method,
annotation instructions, reviewer count, and disagreement resolution.

## Leakage policy

- Keep semantic variants and near duplicates in the same split.
- Never use sealed-test outputs to edit prompts, thresholds, or routing data.
- Record every split's SHA-256 hash in the experiment artifact.
- Version material label corrections instead of silently editing prior data.
- Review medical and legal prompts for personal or sensitive information before
  publishing row-level evidence.

See [the benchmarking protocol](../docs/BENCHMARKING.md) for the required
development/validation/sealed-test workflow and blinded human evaluation.
