#!/usr/bin/env python3
"""P0/P1 physics validation: potential → H → eigenpair → Lanczos → figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.physics.eigensolver import solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian, hamiltonian_symmetry_error
from spectral_krylov_jepa.physics.lanczos import run_lanczos, validate_lanczos
from spectral_krylov_jepa.physics.laplacian import build_laplacian, laplacian_symmetry_error
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.physics.validation import assert_physics_ok
from spectral_krylov_jepa.plotting.physics_figures import plot_physics_panel
from spectral_krylov_jepa.utils.io import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate physics engine")
    parser.add_argument("--n-interior", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    grid = GridSpec(n_interior=args.n_interior)
    out_dir = ensure_dir(args.out_dir)

    print("=== Spectral Krylov-JEPA physics validation ===")
    report = assert_physics_ok(grid=grid, potential_seed=args.seed)
    print(report.message)
    write_json({"ok": report.ok, "checks": report.checks}, out_dir / "physics_validation.json")

    # Detailed single-example figure
    v, _ = generate_potential("id_gaussian_mixture", args.seed, grid=grid)
    ham = build_hamiltonian(grid, v)
    lap_err = laplacian_symmetry_error(build_laplacian(grid))
    ham_err = hamiltonian_symmetry_error(ham)
    print(f"Laplacian symmetry error: {lap_err:.3e}")
    print(f"Hamiltonian symmetry error: {ham_err:.3e}")

    gs = solve_ground_state(ham)
    print(f"E0 = {gs.energy:.8f}, residual_rel = {gs.residual_rel:.3e}")

    lanc = run_lanczos(ham, depth=3, q0_seed=args.seed + 99)
    val = validate_lanczos(ham, lanc)
    print(f"Lanczos validation: {val}")
    if not val["ok"]:
        raise SystemExit("Lanczos validation failed")

    fig_path = plot_physics_panel(
        v, gs.wavefunction, gs.energy, grid, out_dir / "figure_A_physics_panel.png"
    )
    print(f"Wrote figure: {fig_path}")
    print("PHYSICS VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
