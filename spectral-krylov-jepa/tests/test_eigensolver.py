"""Eigensolver tests."""

import numpy as np

from spectral_krylov_jepa.physics.eigensolver import discrete_inner, solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_box_potential, generate_potential


def test_box_ground_state_residual_and_norm():
    g = GridSpec(n_interior=16)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    assert gs.accepted
    assert gs.residual_rel < 1e-5
    assert abs(discrete_inner(gs.wavefunction, gs.wavefunction, g) - 1.0) < 1e-8
    assert gs.energy > 0


def test_gaussian_well_ground_state():
    g = GridSpec(n_interior=16)
    v, _ = generate_potential("id_gaussian_mixture", 7, grid=g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    assert np.isfinite(gs.energy)
    assert gs.residual_rel < 1e-5
