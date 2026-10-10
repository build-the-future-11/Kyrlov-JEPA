"""Analytic amplitude-invariance tests, independent of any protected potential."""

import importlib.util
import os
import sys

import numpy as np
import pytest

from spectral_krylov_jepa.physics import eigensolver
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.evaluation import hybrid_spectral

if os.environ.get("KRYLOV_NORMALIZATION_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "krylov_normalization_predecessor", os.environ["KRYLOV_NORMALIZATION_SOURCE"]
    )
    eigensolver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = eigensolver
    spec.loader.exec_module(eigensolver)


@pytest.mark.parametrize("factor", [1e-250, 1e-160, 1.0, 1e160, 1e250, -1e250])
def test_normalization_preserves_direction_at_arbitrary_finite_amplitude(factor):
    grid = GridSpec(n_interior=3, x_max=2.0, y_max=3.0)
    base = np.arange(1, 10, dtype=float).reshape(3, 3)
    original = base * factor
    actual = eigensolver.normalize_wavefunction(original, grid)
    expected = np.sign(factor) * base / np.sqrt(np.sum(base * base) * cell_area(grid))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=1e-16)
    assert np.sum(actual * actual) * cell_area(grid) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_array_equal(original, base * factor)


def test_ordinary_range_retains_exact_historical_arithmetic():
    grid = GridSpec(n_interior=3)
    psi = np.arange(1, 10, dtype=float).reshape(3, 3)
    expected = psi / np.sqrt(np.sum(psi * psi) * cell_area(grid))
    np.testing.assert_array_equal(eigensolver.normalize_wavefunction(psi, grid), expected)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2)),
        np.full((2, 2), np.nan),
        np.full((2, 2), np.inf),
        np.ones((2, 2), complex) * 1j,
        np.ones((2, 3)),
    ],
)
def test_invalid_wavefunctions_fail_explicitly(bad):
    with pytest.raises(ValueError):
        eigensolver.normalize_wavefunction(bad, GridSpec(n_interior=2))


def test_full_spectral_reconstruction_preserves_coefficient_scale(monkeypatch):
    monkeypatch.setattr(
        hybrid_spectral, "normalize_wavefunction", eigensolver.normalize_wavefunction
    )
    grid = GridSpec(n_interior=4)
    coefficients = np.array([1.0, 0.3, -0.1, 0.2])
    expected = hybrid_spectral.reconstruct_from_coefficients(coefficients, grid, 2)
    actual = hybrid_spectral.reconstruct_from_coefficients(coefficients * 1e250, grid, 2)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=2e-15)
