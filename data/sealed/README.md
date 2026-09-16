# ARCS sealed benchmark registry

## `arcs_sealed_pilot_v1.jsonl`

- Role: sealed pilot evaluation; never use for prompt, router, or threshold tuning.
- Size: 40 prompts, balanced across CODING, MEDICAL, LEGAL, and GENERAL.
- Provenance: newly authored for this repository on 2026-09-16 with AI
  assistance and maintainer review; no prompts were copied from the development
  benchmark or an external proprietary dataset.
- Personal data: none; scenarios are synthetic.
- License: MIT, matching the repository.
- Limitations: author-curated, small, English-only, and not independently
  validated. It is evidence for a pilot comparison, not a general capability
  claim.

The immutable byte hash and pre-run declaration are stored in
`manifest_v1.json`. Any content change requires a new filename and version.
