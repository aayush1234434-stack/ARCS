#!/usr/bin/env bash
# ARCS reproducibility helper — safe subcommands for CI and local smoke checks.
#
# Usage:
#   ./scripts/reproduce.sh check
#   ./scripts/reproduce.sh eval-baseline
#   ./scripts/reproduce.sh rq1-bootstrap
#   ./scripts/reproduce.sh merge
#
# Full pipeline / eval runs require GROQ_API_KEY and NVIDIA_API_KEY; not run here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

FOOTER='Full eval requires GROQ_API_KEY and NVIDIA_API_KEY; not run in CI.'

usage() {
  cat <<EOF
Usage: $(basename "$0") <subcommand>

Subcommands:
  check           Run pytest and a lightweight import smoke test
  eval-baseline   Print baseline eval commands (dry-run plan only; does not call APIs)
  rq1-bootstrap   Build RQ1 bootstrap corpus, prepare datasets, print RQ1 run plan
  merge           Print post-fix merge snapshot plan (dry-run only)

$FOOTER
EOF
}

cmd_check() {
  echo "=== reproduce check ==="
  echo "python: $PYTHON"
  echo

  echo ">>> pytest"
  "$PYTHON" -m pytest tests/ -q

  echo
  echo ">>> import smoke"
  "$PYTHON" - <<'PY'
import arcs
from arcs.eval import compare, experiments, metrics
from arcs.router.classifier import clear_cache, route
from arcs.verification import judge, spec_generator

print("  arcs OK")
print("  eval + router + verification imports OK")
PY

  echo
  echo "$FOOTER"
}

cmd_eval_baseline() {
  echo "=== reproduce eval-baseline (commands only — not executed) ==="
  echo
  cat <<EOF
# Validate held-out eval set
$PYTHON scripts/validate_eval_queries.py

# Snapshot baseline manifest (dry-run plan)
$PYTHON scripts/snapshot_baseline.py --name baseline-v1 --dry-run

# Router eval on frozen test set + eval queries (requires trained router model)
$PYTHON scripts/eval_router.py --name baseline-v1-full-router --eval-queries --skip-train

# Full pipeline eval on data/eval_queries.jsonl (API keys required)
$PYTHON scripts/eval_pipeline.py --name baseline-v1-full --dry-run
$PYTHON scripts/eval_pipeline.py --name baseline-v1-full --sleep-between 2

# Compare after a live run
$PYTHON scripts/compare_experiments.py --list
EOF
  echo
  echo "$FOOTER"
}

cmd_rq1_bootstrap() {
  echo "=== reproduce rq1-bootstrap ==="
  echo

  echo ">>> bootstrap RQ1 corpus"
  "$PYTHON" scripts/bootstrap_rq1_corpus.py

  echo
  echo ">>> prepare RQ1 datasets"
  "$PYTHON" scripts/rq1_prepare_datasets.py

  echo
  echo ">>> RQ1 run plan (dry-run — no training)"
  "$PYTHON" scripts/rq1_run.py --dry-run

  echo
  echo "$FOOTER"
}

cmd_merge() {
  echo "=== reproduce merge (dry-run only) ==="
  echo
  "$PYTHON" scripts/snapshot_post_fix.py --dry-run
  echo
  echo "$FOOTER"
}

SUB="${1:-}"
case "$SUB" in
  check)
    cmd_check
    ;;
  eval-baseline)
    cmd_eval_baseline
    ;;
  rq1-bootstrap)
    cmd_rq1_bootstrap
    ;;
  merge)
    cmd_merge
    ;;
  -h|--help|help|"")
    usage
    [[ -n "$SUB" ]] || exit 0
    ;;
  *)
    echo "Error: unknown subcommand: $SUB" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
