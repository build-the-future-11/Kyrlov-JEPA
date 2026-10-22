"""Decimal-oracle checks for the retained component-square underflow witness."""

from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from spectral_krylov_jepa.physics.eigensolver import normalize_wavefunction
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area


def _oracle(field, grid):
    with localcontext() as context:
        context.prec = 200
        values = [Decimal.from_float(float(value)) for value in field.flat]
        area = Decimal.from_float(cell_area(grid))
        norm = (sum(value * value for value in values) * area).sqrt()
        return np.array([float(value / norm) for value in values]).reshape(field.shape)


def _assert_oracle(field, grid):
    saved = field.copy()
    actual = normalize_wavefunction(field, grid)
    expected = _oracle(field, grid)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(field, saved)
    with localcontext() as context:
        context.prec = 200
        values = [Decimal.from_float(float(value)) for value in actual.flat]
        weighted_square = sum(value * value for value in values) * Decimal.from_float(cell_area(grid))
        assert abs(weighted_square - 1) < Decimal("8e-15")


@pytest.mark.parametrize("spacing_exponent", [-500, 0, 500])
@pytest.mark.parametrize("amplitude", [math.ldexp(1.5, -537), 1e-300, 1e300])
def test_normalization_matches_decimal_across_cell_area_and_amplitude(spacing_exponent, amplitude):
    spacing = math.ldexp(1.0, spacing_exponent)
    grid = GridSpec(2, 0.0, 3 * spacing, 0.0, 3 * spacing)
    field = np.array([[1.0, 2.0], [-1.0, 0.5]]) * amplitude
    _assert_oracle(field, grid)


def test_previous_unexecuted_equal_component_witness():
    spacing = math.ldexp(1.0, 500)
    grid = GridSpec(2, 0.0, 3 * spacing, 0.0, 3 * spacing)
    field = np.full((2, 2), math.ldexp(1.5, -537))
    _assert_oracle(field, grid)
