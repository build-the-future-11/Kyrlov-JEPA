#!/usr/bin/env bash
# Study-2 overnight confirmatory (amendment 0003).
# Fresh v2 data + same-encoder block-JEPA. Does NOT resume Study-1 runs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PKG="$ROOT/research/free-physics-jepa"
LOG="$PKG/runs/overnight_confirmatory_v2.log"

cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$PKG:$PKG/lmop_jepa:$PKG/scripts${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${FREE_PHYSICS_PYTHON:-}" ]]; then
  PY="$FREE_PHYSICS_PYTHON"
elif [[ -x "$ROOT/spectral-krylov-jepa/.venv/bin/python" ]]; then
  PY="$ROOT/spectral-krylov-jepa/.venv/bin/python"
elif [[ -x "$PKG/.venv/bin/python" ]]; then
  PY="$PKG/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

mkdir -p "$PKG/runs"
{
  echo "==== study2 overnight start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "ROOT=$ROOT"
  echo "PYTHON=$PY"
  echo "AMENDMENT=0003_matched_block_jepa"
  "$PY" -c "import torch,h5py,yaml,numpy,matplotlib; print('deps_ok', torch.__version__)"
  if command -v git >/dev/null && [[ -d "$ROOT/.git" ]]; then
    echo "GIT_SHA=$(git -C "$ROOT" rev-parse HEAD)"
  else
    echo "GIT_SHA=NONE"
  fi
  # Optional: pass an existing Study-2 run dir as $1 to resume
  if [[ $# -ge 1 ]]; then
    "$PY" "$PKG/scripts/run_experiment.py" --mode confirmatory --resume-run "$1"
  else
    "$PY" "$PKG/scripts/run_experiment.py" --mode confirmatory
  fi
  echo "==== study2 overnight done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "Artifacts:"
  echo "  $PKG/results/tables/confirmatory_v2.csv"
  echo "  $PKG/results/tables/claim_gate_v2.json"
  echo "  $PKG/figures/confirmatory_v2_label_efficiency.png"
} 2>&1 | tee -a "$LOG"
