# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,optimization]"
pytest -q
```

Live integration tests require the API keys documented in `.env.example`.
Offline tests must not make network calls.

## Pull requests

- Keep generated experiment output out of normal source commits. Publish only
  curated, redacted evidence through `scripts/publish_experiment.py`.
- Add or update tests for behavioral changes.
- Do not weaken the generated-code sandbox or enable the unsafe subprocess
  fallback in deployment configuration.
- Report benchmark changes with the dataset hash, git commit, sample size,
  confidence interval, latency, and error count.
- Never tune against the sealed test split. Use development and validation data.

## Quality checks

```bash
make test
make smoke
make build
```
