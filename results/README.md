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
