#!/usr/bin/env bash
# Overnight confirmatory matrix for LMOP-JEPA.
# Resumes confirmatory_20260926T150845Z (skips completed cells), runs shuffled
# control, finishes remaining finetunes/evals, writes claim gate + paper stub.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PKG="$ROOT/research/free-physics-jepa"
RUN_NAME="${1:-confirmatory_20260926T150845Z}"
LOG="$PKG/runs/overnight_confirmatory.log"

cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$PKG:$PKG/lmop_jepa${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$PKG/runs"
{
  echo "==== overnight start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "ROOT=$ROOT"
  echo "RESUME=$RUN_NAME"
  if command -v git >/dev/null && [[ -d "$ROOT/.git" ]]; then
    echo "GIT_SHA=$(git -C "$ROOT" rev-parse HEAD)"
  else
    echo "GIT_SHA=NONE"
  fi
  python3 "$PKG/scripts/run_experiment.py" \
    --mode confirmatory \
    --resume-run "$RUN_NAME"
  echo "==== overnight done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "Artifacts:"
  echo "  $PKG/results/tables/confirmatory.csv"
  echo "  $PKG/results/tables/claim_gate.json"
  echo "  $PKG/results/tables/claim_gate.md"
  echo "  $PKG/paper/RESULTS_AUTO.md"
  echo "  $PKG/figures/confirmatory_label_efficiency.png"
} 2>&1 | tee -a "$LOG"
