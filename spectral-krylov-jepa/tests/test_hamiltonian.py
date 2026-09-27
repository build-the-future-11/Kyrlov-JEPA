"""Hamiltonian tests."""

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian, hamiltonian_symmetry_error
from spectral_krylov_jepa.physics.potentials import generate_box_potential, generate_potential


def test_hamiltonian_symmetry():
    g = GridSpec(n_interior=16)
    v, _ = generate_potential("id_gaussian_mixture", 123, grid=g)
    ham = build_hamiltonian(g, v)
    assert hamiltonian_symmetry_error(ham) < 1e-12
    assert ham.matrix.shape == (g.n_dof, g.n_dof)


def test_box_hamiltonian_positive_definite_kinetic():
    g = GridSpec(n_interior=12)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    dens = ham.to_dense()
    evals = np.linalg.eigvalsh(dens)
    assert evals[0] > 0
