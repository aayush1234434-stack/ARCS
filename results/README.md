# Auditable result bundles

This directory is for small, redacted experiment bundles that reviewers can
inspect without access to local API keys or ignored artifacts.

Publish a completed local run with:

```bash
python scripts/publish_experiment.py artifacts/experiments/<run-id>
```

Each bundle contains:

- `experiment.json` — machine-readable metrics and, when available, row-level evidence
- `report.md` — human-readable metrics, confidence interval, provenance, and limits
- `manifest.json` — SHA-256 checksums and an explicit evidence-level declaration

Do not publish personal data, proprietary prompts, credentials, or raw user
feedback without consent. A summary-only bundle is not sufficient evidence for
a comparative performance claim.

## Current sealed pilot status

The published 20B bundles are an audit trail for a provider-limited pilot, not
a final three-way benchmark claim. The naive and no-verification conditions
completed all 40 sealed rows. The full ARCS condition stopped after 22 rows
(21 completed plus one quota error) when Groq's 200,000-token daily limit was
reached. The matched blinded pilot packet therefore uses the seven rows for
which all three systems produced answers; see the ignored local artifact at
`artifacts/human-eval/sealed-v1-gpt20b-common7/`.

Do not report the partial full-ARCS PASS rate as a result for the complete
40-row sealed set. A final comparative claim requires completing the remaining
full-ARCS rows under the same frozen configuration, then collecting ratings
from at least two independent reviewers.
