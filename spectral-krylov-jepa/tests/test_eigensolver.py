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


def test_explicit_start_vector_repeats_exactly():
    g = GridSpec(n_interior=12)
    v, _ = generate_potential("id_gaussian_mixture", 17, grid=g)
    ham = build_hamiltonian(g, v)
    v0 = np.linspace(1.0, 2.0, g.n_dof, dtype=np.float64)

    first = solve_ground_state(ham, v0=v0)
    second = solve_ground_state(ham, v0=v0)

    assert first.energy == second.energy
    assert np.array_equal(first.wavefunction, second.wavefunction)
    assert first.residual_norm == second.residual_norm


def test_invalid_start_vector_fails_closed():
    g = GridSpec(n_interior=8)
    v, _ = generate_potential("id_gaussian_mixture", 19, grid=g)
    ham = build_hamiltonian(g, v)

    with np.testing.assert_raises_regex(ValueError, "expected"):
        solve_ground_state(ham, v0=np.ones(g.n_dof - 1))
    with np.testing.assert_raises_regex(ValueError, "finite"):
        bad = np.ones(g.n_dof)
        bad[0] = np.nan
        solve_ground_state(ham, v0=bad)
    with np.testing.assert_raises_regex(ValueError, "nonzero"):
        solve_ground_state(ham, v0=np.zeros(g.n_dof))
