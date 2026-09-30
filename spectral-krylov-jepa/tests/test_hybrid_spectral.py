"""Tests for the adaptive perturbation--Ritz hybrid solver."""

import numpy as np

from spectral_krylov_jepa.evaluation.hybrid_spectral import (
    adaptive_ritz,
    first_order_perturbation_fast,
    first_second_order_perturbation,
    box_energies,
    low_mode_mask,
    projected_ritz,
    sine_basis,
    spectral_davidson_direction,
    residual_expansion_direction,
    variational_monotonicity_gap,
)
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_box_potential, generate_potential


def test_free_box_perturbation_has_no_excited_coefficients():
    grid = GridSpec(n_interior=12)
    v, _ = generate_box_potential(grid)
    c1, c2, _, _ = first_second_order_perturbation(v, grid, side=5)
    assert abs(c1[0] - 1.0) < 1e-12
    assert abs(c2[0] - 1.0) < 1e-12
    assert np.linalg.norm(c1[1:]) < 1e-12
    assert np.linalg.norm(c2[1:]) < 1e-12


def test_adaptive_basis_cannot_raise_ritz_ground_energy():
    grid = GridSpec(n_interior=12)
    v, _ = generate_potential("id_gaussian_mixture", 1234, grid=grid)
    b3, _ = sine_basis(grid, 3)
    small = projected_ritz(v, grid, b3)

    rng = np.random.default_rng(7)
    proposal = rng.normal(size=grid.n_dof)
    large = adaptive_ritz(
        v,
        grid,
        low_side=3,
        proposal_vectors=[proposal],
    )
    assert large.basis_dim == 10
    assert large.proposal_rank == 1
    assert variational_monotonicity_gap(small, large) >= -1e-9


def test_low_mode_mask_selects_square_block():
    mask = low_mode_mask(side=7, low_side=3)
    assert mask.shape == (49,)
    assert int(mask.sum()) == 9


def test_fast_first_order_matches_full_first_order():
    grid = GridSpec(n_interior=12)
    v, _ = generate_potential("id_gaussian_mixture", 5678, grid=grid)
    c_fast, e_fast = first_order_perturbation_fast(v, grid, side=5)
    c_full, _, e_full, _ = first_second_order_perturbation(v, grid, side=5)
    assert np.allclose(c_fast, c_full, atol=1e-11, rtol=1e-11)
    assert abs(e_fast - e_full) < 1e-11


def test_analytic_box_energies_match_discrete_hamiltonian_projection():
    grid = GridSpec(n_interior=12)
    basis, _ = sine_basis(grid, 5)
    ham = build_hamiltonian(grid, np.zeros((grid.ny, grid.nx)))
    projected = basis.T @ (ham.matrix @ basis)
    assert np.allclose(
        np.diag(projected),
        box_energies(grid, 5),
        atol=1e-10,
        rtol=1e-10,
    )
    offdiag = projected - np.diag(np.diag(projected))
    assert np.max(np.abs(offdiag)) < 1e-9


def test_davidson_and_residual_directions_expand_low_ritz_space():
    grid = GridSpec(n_interior=12)
    v, _ = generate_potential("id_gaussian_mixture", 9012, grid=grid)
    low_basis, _ = sine_basis(grid, 3)
    low = projected_ritz(v, grid, low_basis, assume_orthonormal=True)

    for builder in [spectral_davidson_direction, residual_expansion_direction]:
        direction, _ = builder(v, grid, low_side=3)
        expanded = adaptive_ritz(
            v,
            grid,
            low_side=3,
            proposal_vectors=[direction],
        )
        assert expanded.basis_dim == 10
        assert expanded.energy <= low.energy + 1e-9
