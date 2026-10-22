"""Independent span, Rayleigh quotient, and orthogonality regressions."""

import numpy as np
import pytest
from scipy.linalg import eigh

from spectral_krylov_jepa.evaluation.hybrid_spectral import (
    adaptive_ritz,
    projected_ritz,
    sine_basis,
)
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


def problem():
    grid = GridSpec(n_interior=6)
    potential = np.linspace(-1.0, 2.0, grid.n_dof).reshape(grid.ny, grid.nx)
    return grid, potential, build_hamiltonian(grid, potential).matrix.toarray()


@pytest.mark.parametrize("columns", ["duplicate", "zero_first", "zero_last", "dependent"])
def test_dependent_columns_do_not_invent_trial_directions(columns):
    grid, potential, operator = problem()
    vector = np.random.default_rng(19).normal(size=grid.n_dof)
    choices = {
        "duplicate": [vector, vector],
        "zero_first": [np.zeros_like(vector), vector],
        "zero_last": [vector, np.zeros_like(vector)],
        "dependent": [vector, 2 * vector, -3 * vector],
    }
    result = projected_ritz(potential, grid, np.column_stack(choices[columns]))
    expected = float(vector @ operator @ vector / (vector @ vector))
    assert result.basis_dim == 1
    assert result.energy == pytest.approx(expected, rel=1e-12)
    wave = result.wavefunction.ravel()
    residual = wave - vector * (vector @ wave) / (vector @ vector)
    assert np.linalg.norm(residual) < 1e-11


def test_full_rank_result_matches_generalized_eigenproblem():
    grid, potential, operator = problem()
    basis = np.random.default_rng(20).normal(size=(grid.n_dof, 4))
    expected = eigh(basis.T @ operator @ basis, basis.T @ basis)[0][0]
    result = projected_ritz(potential, grid, basis)
    assert result.basis_dim == 4
    assert result.energy == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-250, -3.0, 1e250])
def test_global_basis_scale_and_column_order_preserve_subspace(scale):
    grid, potential, _ = problem()
    basis = np.random.default_rng(21).normal(size=(grid.n_dof, 3))
    expected = projected_ritz(potential, grid, basis)
    actual = projected_ritz(potential, grid, scale * basis[:, ::-1])
    assert actual.basis_dim == expected.basis_dim
    assert actual.energy == pytest.approx(expected.energy, rel=1e-12)


@pytest.mark.parametrize("kind", ["empty", "zero", "nan", "infinity", "complex"])
def test_invalid_basis_does_not_produce_an_eigenpair(kind):
    grid, potential, _ = problem()
    basis = np.ones((grid.n_dof, 1))
    if kind == "empty":
        basis = basis[:, :0]
    elif kind == "zero":
        basis[:] = 0
    elif kind == "nan":
        basis[0, 0] = np.nan
    elif kind == "infinity":
        basis[0, 0] = np.inf
    else:
        basis = basis.astype(complex) + 1j
    with pytest.raises(ValueError):
        projected_ritz(potential, grid, basis)


def test_orthonormal_fast_path_checks_its_mathematical_assumption():
    grid, potential, _ = problem()
    basis, _ = sine_basis(grid, 2)
    with pytest.raises(ValueError, match="orthonormal"):
        projected_ritz(potential, grid, 2 * basis, assume_orthonormal=True)
    valid = projected_ritz(potential, grid, basis, assume_orthonormal=True)
    general = projected_ritz(potential, grid, basis)
    assert valid.energy == pytest.approx(general.energy, rel=1e-12)


def test_adaptive_reorthogonalizes_low_mode_contamination():
    grid, potential, operator = problem()
    low, _ = sine_basis(grid, 1)
    raw = np.random.default_rng(23).normal(size=grid.n_dof)
    proposal = 1e12 * low[:, 0] + raw
    # The independent oracle uses the actual representable proposal's span.
    residual = proposal - low[:, 0] * (low[:, 0] @ proposal)
    residual -= low[:, 0] * (low[:, 0] @ residual)
    basis = np.column_stack([low, residual / np.linalg.norm(residual)])
    expected = eigh(basis.T @ operator @ basis, basis.T @ basis)[0][0]
    actual = adaptive_ritz(potential, grid, low_side=1, proposal_vectors=[proposal])
    assert actual.basis_dim == 2
    assert actual.proposal_rank == 1
    assert actual.energy == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("tol", [0.0, -1.0, np.nan, np.inf])
def test_adaptive_tolerance_must_be_finite_positive(tol):
    grid, potential, _ = problem()
    with pytest.raises(ValueError, match="orthogonal_tol"):
        adaptive_ritz(potential, grid, low_side=1, orthogonal_tol=tol)


def test_adaptive_rejects_nonfinite_proposals():
    grid, potential, _ = problem()
    with pytest.raises(ValueError, match="finite"):
        adaptive_ritz(potential, grid, proposal_vectors=[np.full(grid.n_dof, np.nan)])
