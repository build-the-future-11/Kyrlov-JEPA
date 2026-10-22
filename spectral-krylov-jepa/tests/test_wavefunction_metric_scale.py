"""Closed-form finite-difference fixtures; no data or eigensolver run."""

import numpy as np
import pytest

from spectral_krylov_jepa.evaluation.metrics import (
    evaluate_example,
    rayleigh_quotient,
    schrodinger_residual,
)
from spectral_krylov_jepa.physics.grid import GridSpec


@pytest.fixture
def analytic_case():
    # On a 2x2 interior unit-square grid, h=1/3. H has diagonal 18+V
    # and nearest-neighbour off-diagonals -4.5. For psi=e_0 and V=0:
    # Rayleigh=18; at E=17, residual=sqrt(1 + 4.5^2 + 4.5^2).
    return GridSpec(n_interior=2), np.zeros((2, 2)), np.array([[1., 0.], [0., 0.]])


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-20, 1., -3., 1e150, 1e300])
def test_rayleigh_is_invariant_to_state_amplitude(analytic_case, scale):
    grid, potential, state = analytic_case
    assert rayleigh_quotient(potential, state * scale, grid) == pytest.approx(18.)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-20, 1., -3., 1e150, 1e300])
def test_relative_residual_is_invariant_to_state_amplitude(analytic_case, scale):
    grid, potential, state = analytic_case
    expected = np.sqrt(41.5)
    assert schrodinger_residual(potential, state * scale, 17., grid) == pytest.approx(expected)


def test_exact_constant_box_mode_has_known_energy_without_an_eigensolver(analytic_case):
    grid, potential, _ = analytic_case
    # [1,1,1,1] is an exact eigenvector of this 2x2-grid Hamiltonian, E=9.
    state = np.ones((2, 2)) * 1e-250
    assert rayleigh_quotient(potential, state, grid) == pytest.approx(9.)
    assert schrodinger_residual(potential, state, 9., grid) == pytest.approx(0., abs=1e-14)
    assert schrodinger_residual(potential, state, 12., grid) == pytest.approx(3.)


def test_shifted_potential_and_energy_preserve_residual(analytic_case):
    grid, potential, state = analytic_case
    assert rayleigh_quotient(potential + 2.5, state, grid) == pytest.approx(20.5)
    assert schrodinger_residual(potential + 2.5, state, 19.5, grid) == pytest.approx(np.sqrt(41.5))


def test_rayleigh_reduction_handles_representable_extreme_energy(analytic_case):
    grid, potential, state = analytic_case
    assert rayleigh_quotient(potential + 1e308, np.ones_like(state), grid) == pytest.approx(1e308)


def test_residual_norm_handles_representable_extreme_energy(analytic_case):
    grid, potential, state = analytic_case
    assert schrodinger_residual(potential, state, 1e300, grid) == pytest.approx(1e300)


@pytest.mark.parametrize("metric", [rayleigh_quotient, schrodinger_residual])
@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "complex", "size"])
def test_invalid_states_do_not_become_successful_metrics(analytic_case, metric, kind):
    grid, potential, state = analytic_case
    invalid = {
        "zero": np.zeros((2, 2)),
        "nan": np.full((2, 2), np.nan),
        "inf": np.full((2, 2), np.inf),
        "complex": state.astype(np.complex128) + 1j,
        "size": np.ones(3),
    }[kind]
    args = (potential, invalid, 17., grid) if metric is schrodinger_residual else (potential, invalid, grid)
    with pytest.raises(ValueError):
        metric(*args)


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), -float("inf")])
def test_residual_rejects_nonfinite_energy(analytic_case, energy):
    grid, potential, state = analytic_case
    with pytest.raises(ValueError, match="energy"):
        schrodinger_residual(potential, state, energy, grid)


def test_full_evaluator_rejects_zero_prediction_before_emitting_false_zero_residual(analytic_case):
    grid, potential, state = analytic_case
    with pytest.raises(ValueError, match="nonzero"):
        evaluate_example(potential, state, 18., np.zeros_like(state), 18., grid)


def test_flattened_and_grid_states_agree_without_mutation(analytic_case):
    grid, potential, state = analytic_case
    saved = state.copy()
    assert rayleigh_quotient(potential, state.ravel(), grid) == rayleigh_quotient(potential, state, grid)
    assert schrodinger_residual(potential, state.ravel(), 17., grid) == schrodinger_residual(potential, state, 17., grid)
    np.testing.assert_array_equal(state, saved)
