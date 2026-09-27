#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
mkdir -p .astra/tmp .astra/mpl .astra/cache
export TMPDIR="$ROOT/.astra/tmp" MPLCONFIGDIR="$ROOT/.astra/mpl" XDG_CACHE_HOME="$ROOT/.astra/cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$ROOT/spectral-krylov-jepa/.venv/bin/python"
"$PY" -m pytest spectral-krylov-jepa/tests tests -q --basetemp="$ROOT/.astra/pytest-local"
"$PY" -m compileall -q spectral-krylov-jepa/src spectral-krylov-jepa/scripts
ruff check spectral-krylov-jepa/src spectral-krylov-jepa/scripts --select E9,F63,F7,F82
"$PY" spectral-krylov-jepa/scripts/00_validate_physics.py --out-dir .astra/physics
printf '%s\n' 'Engineering checks passed. This does not establish scientific efficacy or submission readiness.'
