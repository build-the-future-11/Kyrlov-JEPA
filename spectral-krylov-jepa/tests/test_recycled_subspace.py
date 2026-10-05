"""Tests for the budget-matched recycled Ritz classical control."""

import numpy as np
import pytest

from spectral_krylov_jepa.evaluation.hybrid_spectral import projected_ritz
from spectral_krylov_jepa.evaluation.recycled_subspace import (
    build_recycle_space,
    solve_recycled_ritz_from_potential,
)
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_potential


def _prior_vectors(grid: GridSpec) -> list[np.ndarray]:
    out = []
    for seed in (101, 202, 303):
        potential, _ = generate_potential("id_gaussian_mixture", seed, grid=grid)
        result = solve_ground_state(build_hamiltonian(grid, potential))
        out.append(result.wavefunction.reshape(-1))
    return out


def test_build_recycle_space_is_fixed_rank_and_orthonormal():
    grid = GridSpec(n_interior=10)
    basis = build_recycle_space(_prior_vectors(grid), n_dof=grid.n_dof, rank=2)
    assert basis.shape == (grid.n_dof, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-10, rtol=1e-10)


def test_recycle_space_fails_closed_if_rank_is_not_available():
    grid = GridSpec(n_interior=8)
    v = np.ones(grid.n_dof)
    with pytest.raises(ValueError, match="only 1 independent"):
        build_recycle_space([v, 2.0 * v], n_dof=grid.n_dof, rank=2)


def test_recycled_solver_recomputes_new_operator_action_and_matches_projected_ritz():
    grid = GridSpec(n_interior=10)
    basis = build_recycle_space(_prior_vectors(grid), n_dof=grid.n_dof, rank=2)
    v1, _ = generate_potential("id_gaussian_mixture", 404, grid=grid)
    v2, _ = generate_potential("id_gaussian_mixture", 505, grid=grid)

    r1 = solve_recycled_ritz_from_potential(v1, grid, basis, refresh_threshold=1.0)
    r2 = solve_recycled_ritz_from_potential(v2, grid, basis, refresh_threshold=1.0)

    direct = projected_ritz(v1, grid, basis, assume_orthonormal=True)
    assert abs(r1.energy - direct.energy) < 1e-10
    assert r1.budget.operator_applications == 2
    assert r1.budget.basis_rank == 2
    assert abs(r1.energy - r2.energy) > 1e-8


def test_refresh_rule_is_explicit_and_threshold_driven():
    grid = GridSpec(n_interior=10)
    basis = build_recycle_space(_prior_vectors(grid), n_dof=grid.n_dof, rank=1)
    potential, _ = generate_potential("id_gaussian_mixture", 606, grid=grid)

    probe = solve_recycled_ritz_from_potential(
        potential, grid, basis, refresh_threshold=1.0
    )
    assert np.isfinite(probe.residual_relative_to_hx)
    assert probe.residual_relative_to_hx >= 0.0

    threshold_low = max(0.0, probe.residual_relative_to_hx * 0.5)
    forced_refresh = solve_recycled_ritz_from_potential(
        potential, grid, basis, refresh_threshold=threshold_low
    )
    no_refresh = solve_recycled_ritz_from_potential(
        potential,
        grid,
        basis,
        refresh_threshold=probe.residual_relative_to_hx + 1e-6,
    )
    if probe.residual_relative_to_hx == 0.0:
        rng = np.random.default_rng(7)
        random_basis = build_recycle_space(
            [rng.normal(size=grid.n_dof)], n_dof=grid.n_dof, rank=1
        )
        forced_refresh = solve_recycled_ritz_from_potential(
            potential, grid, random_basis, refresh_threshold=0.0
        )
    assert forced_refresh.refresh_requested
    assert not no_refresh.refresh_requested


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf])
def test_refresh_threshold_must_be_predeclared_finite_nonnegative(threshold):
    grid = GridSpec(n_interior=8)
    basis = build_recycle_space(_prior_vectors(grid), n_dof=grid.n_dof, rank=1)
    potential, _ = generate_potential("id_gaussian_mixture", 707, grid=grid)
    with pytest.raises(ValueError, match="refresh_threshold"):
        solve_recycled_ritz_from_potential(
            potential, grid, basis, refresh_threshold=threshold
        )


def test_nonorthonormal_basis_is_rejected_instead_of_silently_changed():
    grid = GridSpec(n_interior=8)
    potential, _ = generate_potential("id_gaussian_mixture", 808, grid=grid)
    bad = np.column_stack([np.ones(grid.n_dof), np.ones(grid.n_dof)])
    with pytest.raises(ValueError, match="orthonormal"):
        solve_recycled_ritz_from_potential(
            potential, grid, bad, refresh_threshold=0.1
        )
