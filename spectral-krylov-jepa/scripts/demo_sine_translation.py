"""Save analytic translated-grid Ritz fixtures; no trained/protected examples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spectral_krylov_jepa.evaluation.hybrid_spectral import adaptive_ritz, sine_basis
from spectral_krylov_jepa.physics.grid import GridSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a retained fixture")
    cases = []
    for origin in [0.0, 1e16]:
        grid = GridSpec(8, origin, origin + 4.0, -origin, -origin + 8.0)
        potential = np.zeros((8, 8))
        basis, labels = sine_basis(grid, 3)
        result = adaptive_ritz(potential, grid, low_side=3)
        cases.append({"grid": grid.to_dict(), "potential": potential.tolist(), "basis": basis.tolist(), "labels": labels.tolist(), "energy": result.energy, "wavefunction": result.wavefunction.tolist(), "basis_dim": result.basis_dim, "gram_max_error": float(np.max(np.abs(basis.T @ basis - np.eye(9))))})
    root = Path(__file__).resolve().parents[2]
    paths = ["spectral-krylov-jepa/src/spectral_krylov_jepa/evaluation/hybrid_spectral.py", "spectral-krylov-jepa/src/spectral_krylov_jepa/physics/grid.py", "spectral-krylov-jepa/src/spectral_krylov_jepa/physics/hamiltonian.py", "spectral-krylov-jepa/src/spectral_krylov_jepa/physics/eigensolver.py", "spectral-krylov-jepa/scripts/demo_sine_translation.py"]
    report = {"scope": "analytic classical numerical fixture only", "cases": cases, "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(cases), "energies": [case["energy"] for case in cases], "output": str(args.output)}))


if __name__ == "__main__":
    main()
