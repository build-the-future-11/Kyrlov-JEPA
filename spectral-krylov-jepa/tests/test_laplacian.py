"""Laplacian tests."""

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.laplacian import build_laplacian, laplacian_symmetry_error


def test_laplacian_shape_and_symmetry():
    g = GridSpec(n_interior=16)
    lap = build_laplacian(g)
    assert lap.shape == (g.n_dof, g.n_dof)
    assert laplacian_symmetry_error(lap) < 1e-12


def test_laplacian_negative_semidefinite_ish():
    """Dirichlet Laplacian eigenvalues should be <= 0."""
    g = GridSpec(n_interior=8)
    lap = build_laplacian(g).toarray()
    evals = np.linalg.eigvalsh(lap)
    assert np.all(evals <= 1e-8)
