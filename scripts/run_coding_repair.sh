#!/usr/bin/env bash
# CODING specialist repair: extract queue → DSPy optimize → apply sidecar → re-eval hint.
#
# Usage:
#   ./scripts/run_coding_repair.sh            # full path (optimize needs GROQ_API_KEY)
#   ./scripts/run_coding_repair.sh --dry-run  # plan + optimize/apply dry-runs only
#
# Sidecar output: artifacts/prompts/coding_optimized.txt
# Target module:  arcs/pipelines/specialists/coding.py (+ .bak on apply)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

SIDECAR="artifacts/prompts/coding_optimized.txt"
TARGET="arcs/pipelines/specialists/coding.py"

echo "=== CODING specialist repair ==="

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[1/3] repair.py — extract SPECIALIST queue (dry-run plan)"
  python scripts/repair.py --dry-run --component SPECIALIST --domain CODING
else
  echo "[1/3] repair.py — extract SPECIALIST queue"
  python scripts/repair.py --component SPECIALIST --domain CODING
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[2/3] optimize_coding.py — dry-run (no GROQ / no sidecar write)"
  python scripts/optimize_coding.py --dry-run --max-examples 20
else
  if [[ ! -f "$SIDECAR" ]]; then
    echo "[2/3] optimize_coding.py — generating $SIDECAR (requires GROQ_API_KEY)"
    python scripts/optimize_coding.py --max-examples 20
  else
    echo "[2/3] optimize_coding.py — skipped ($SIDECAR already exists; delete to re-optimize)"
  fi
fi

if [[ ! -f "$SIDECAR" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[3/3] apply_sidecar.py — skipped ($SIDECAR missing)"
    echo "After optimize, preview with:"
    echo "  python scripts/apply_sidecar.py --prompt $SIDECAR --target $TARGET --dry-run"
  else
    echo "Sidecar missing: $SIDECAR"
    echo "Run: python scripts/optimize_coding.py"
    echo "Then: python scripts/apply_sidecar.py --prompt $SIDECAR --target $TARGET --dry-run"
    exit 1
  fi
elif [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[3/3] apply_sidecar.py — dry-run (unified diff, no write)"
  python scripts/apply_sidecar.py --prompt "$SIDECAR" --target "$TARGET" --dry-run
else
  echo "[3/3] apply_sidecar.py — apply with backup ($TARGET.bak)"
  python scripts/apply_sidecar.py --prompt "$SIDECAR" --target "$TARGET"
fi

echo ""
echo "re-run: python scripts/eval_pipeline.py --execute --domains CODING --name post-fix-coding-v4"
