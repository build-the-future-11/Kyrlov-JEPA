"""Physics validation helpers used by scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectral_krylov_jepa.physics.eigensolver import (
    discrete_inner,
    residual_stats,
    solve_ground_state,
)
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import (
    build_hamiltonian,
    hamiltonian_symmetry_error,
)
from spectral_krylov_jepa.physics.lanczos import run_lanczos, validate_lanczos
from spectral_krylov_jepa.physics.laplacian import build_laplacian, laplacian_symmetry_error
from spectral_krylov_jepa.physics.potentials import generate_box_potential, generate_potential


@dataclass
class PhysicsValidationReport:
    ok: bool
    checks: dict[str, Any]
    energy: float | None = None
    residual_rel: float | None = None
    message: str = ""


def validate_physics_pipeline(
    *,
    grid: GridSpec | None = None,
    potential_seed: int = 42,
    family: str = "id_gaussian_mixture",
    symmetry_tol: float = 1e-12,
    residual_tol: float = 1e-5,
    lanczos_depth: int = 3,
) -> PhysicsValidationReport:
    """Run the full P0/P1 physics sanity pipeline and return a report."""
    grid = grid or GridSpec(n_interior=32)
    checks: dict[str, Any] = {}

    lap = build_laplacian(grid)
    checks["laplacian_shape"] = lap.shape == (grid.n_dof, grid.n_dof)
    checks["laplacian_symmetry_error"] = laplacian_symmetry_error(lap)
    checks["laplacian_symmetric"] = checks["laplacian_symmetry_error"] <= symmetry_tol

    v, spec = generate_potential(family, potential_seed, grid=grid)
    checks["potential_finite"] = bool(np.all(np.isfinite(v)))
    checks["potential_family"] = spec.family

    ham = build_hamiltonian(grid, v)
    checks["hamiltonian_symmetry_error"] = hamiltonian_symmetry_error(ham)
    checks["hamiltonian_symmetric"] = checks["hamiltonian_symmetry_error"] <= symmetry_tol

    # Box sanity: lowest mode should be positive and residual-small
    v_box, _ = generate_box_potential(grid)
    ham_box = build_hamiltonian(grid, v_box)
    box_gs = solve_ground_state(ham_box, residual_tol=residual_tol)
    checks["box_energy_positive"] = box_gs.energy > 0
    checks["box_residual_rel"] = box_gs.residual_rel
    checks["box_normalized"] = abs(discrete_inner(box_gs.wavefunction, box_gs.wavefunction, grid) - 1.0) < 1e-8

    gs = solve_ground_state(ham, residual_tol=residual_tol)
    checks["ground_residual_rel"] = gs.residual_rel
    checks["ground_accepted"] = gs.accepted
    checks["ground_energy"] = gs.energy

    lanc = run_lanczos(ham, depth=lanczos_depth, q0_seed=potential_seed + 99)
    lanc_val = validate_lanczos(ham, lanc)
    checks["lanczos"] = lanc_val

    ok = all(
        [
            checks["laplacian_shape"],
            checks["laplacian_symmetric"],
            checks["potential_finite"],
            checks["hamiltonian_symmetric"],
            checks["box_energy_positive"],
            checks["box_normalized"],
            checks["ground_accepted"],
            lanc_val["ok"],
        ]
    )
    msg = "Physics validation passed" if ok else "Physics validation FAILED"
    return PhysicsValidationReport(
        ok=ok,
        checks=checks,
        energy=gs.energy,
        residual_rel=gs.residual_rel,
        message=msg,
    )


def assert_physics_ok(**kwargs: Any) -> PhysicsValidationReport:
    report = validate_physics_pipeline(**kwargs)
    if not report.ok:
        raise AssertionError(f"{report.message}: {report.checks}")
    return report
