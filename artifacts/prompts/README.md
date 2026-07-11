# DSPy sidecar prompts

Sidecar files are **review-only** outputs from `scripts/optimize_*.py`. They are not committed by default (local artifacts). Apply to source with `scripts/apply_sidecar.py` after human review.

## CODING specialist

If `coding_optimized.txt` is missing, generate it from the specialist failure queue:

```bash
# 1. Ensure queues exist (demo/batch failures + extract)
python scripts/repair.py --component SPECIALIST --domain CODING --dry-run

# 2. Generate sidecar (requires GROQ_API_KEY; default --breadth 2 --depth 2)
python scripts/optimize_coding.py --breadth 2 --depth 2

# 3. Preview diff before applying (creates coding.py.bak on real apply)
python scripts/apply_sidecar.py \
  --prompt artifacts/prompts/coding_optimized.txt \
  --target arcs/pipelines/specialists/coding.py \
  --dry-run

# 4. Apply after review
python scripts/apply_sidecar.py \
  --prompt artifacts/prompts/coding_optimized.txt \
  --target arcs/pipelines/specialists/coding.py
```

Or run the full scripted path:

```bash
./scripts/run_coding_repair.sh          # extract + optimize + apply
./scripts/run_coding_repair.sh --dry-run  # plan only; no API optimize, no writes
```

After apply, re-evaluate CODING:

```bash
python scripts/eval_pipeline.py --execute --domains CODING --name post-fix-coding-v4
```
