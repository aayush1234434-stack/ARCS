# Local experiment artifacts

This directory holds **local eval run outputs**. Contents are **gitignored** except this README.

## Layout

Each run is a timestamped folder:

```
artifacts/experiments/2026-07-11T12-36-52_post-fix-v2-merged/
├── experiment.json   # router + pipeline metrics and per-row results
└── summary.txt       # human-readable snapshot (when generated)
```

Naming pattern: `{ISO-timestamp}_{slug}` (e.g. `baseline-v1-full-pipeline`, `post-fix-legal-v1`, `rq1-run-a`).

## Key runs (reference for docs)

| Slug | Role |
|---|---|
| `2026-07-10T07-24-20_baseline-v1-full-pipeline` | Pre-repair end-to-end baseline |
| `2026-07-11T12-36-52_post-fix-v2-merged` | Post-fix FINAL (merged domain runs + resume) |
| `2026-07-11T08-38-52_rq1/` | RQ1 manifest (Run A vs Run B) |
| `2026-07-11T11-22-52_repair_ablation/` | RQ1-bis router ablation |

Numbers in [docs/RESULTS.md](../../docs/RESULTS.md) and [docs/SCORECARD.md](../../docs/SCORECARD.md) cite these paths.

## How to generate

```bash
# Full pipeline eval (requires GROQ_API_KEY + NVIDIA_API_KEY in .env)
python scripts/eval_pipeline.py --execute --name my-run

# Baseline snapshot (no API calls if artifacts exist)
python scripts/snapshot_baseline.py

# Post-fix merge (no API calls; merges latest domain runs)
python scripts/snapshot_post_fix.py

# RQ1 bootstrap
python scripts/rq1_run.py --execute
```

Resume after rate limits:

```bash
python scripts/eval_pipeline.py --execute --resume-from artifacts/experiments/<partial-run>/
```

## Sharing / thesis bundle

Because runs are not committed, export checksums or copy `experiment.json` + `manifest.json` into your thesis appendix or a release tarball. Regenerate locally with `scripts/reproduce.sh` where possible.
