# Benchmarking protocol

ARCS separates development decisions from final claims. The existing 48-query
set is a development benchmark because it has already informed prompt and
pipeline changes; it must not be described as a sealed test set.

## Required splits

- **Development:** inspect failures and change prompts, routing, or verification.
- **Validation:** select thresholds and configurations; do not report it as final.
- **Sealed test:** evaluate once after configuration is frozen. Target at least
  200–500 stratified queries and preserve its SHA-256 fingerprint.

Keep near-duplicate prompts and variants in the same split to prevent leakage.

## Minimum comparison

Run the naive baseline, ARCS without verification, and full ARCS with identical
generator access. Report completed PASS rate, error rate, 95% Wilson interval,
median and p95 latency, measured token usage, calls per query, and per-domain
results. Repeat stochastic runs at least three times.

## Blinded human review

Create a randomized packet from saved experiments:

```bash
python scripts/prepare_human_eval.py \
  --system naive=artifacts/experiments/<naive-run> \
  --system arcs=artifacts/experiments/<arcs-run>
```

Give `review_packet.csv` to at least two reviewers. Keep `answer_key.json` hidden
until annotations are frozen. Reviewers score correctness, completeness, safety,
and citation quality from 1–5 and separately flag harmful answers. Resolve neither
disagreements nor missing ratings before calculating inter-rater agreement.

## Publishing evidence

```bash
python scripts/publish_experiment.py artifacts/experiments/<run>
```

Commit the generated bundle under `results/` only after checking it for personal
data, proprietary content, and secrets. The report explicitly distinguishes an
LLM-judge proxy from human evaluation.
