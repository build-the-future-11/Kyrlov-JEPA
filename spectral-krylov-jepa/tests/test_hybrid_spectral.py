"""Tests for the adaptive perturbation--Ritz hybrid solver."""

import numpy as np

from spectral_krylov_jepa.evaluation.hybrid_spectral import (
    adaptive_ritz,
    first_second_order_perturbation,
    low_mode_mask,
    projected_ritz,
    sine_basis,
    variational_monotonicity_gap,
)
from spectral_krylov_jepa.physics.grid import GridSpec
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
