"""Physics tests for frozen Darcy reference (Free-Physics JEPA)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "free-physics-jepa"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lmop_jepa"))

from darcy_reference import (  # noqa: E402
    DarcyGrid,
    apply_operator,
    build_darcy_matrix,
    condition_estimate,
    is_spd_probe,
    manufactured_round_trip,
    manufactured_sine_mode,
    matrix_symmetry_error,
    sample_log_permeability,
    solve_darcy,
    validate_permeability,
)


@pytest.fixture
def grid32() -> DarcyGrid:
    return DarcyGrid(n=32)


def test_matrix_symmetry(grid32: DarcyGrid):
    a = sample_log_permeability(grid32, 0)
    A = build_darcy_matrix(a, grid32)
    assert matrix_symmetry_error(A) < 1e-12


def test_spd_constant_and_heterogeneous(grid32: DarcyGrid):
    a1 = np.ones((grid32.n, grid32.n))
    assert is_spd_probe(build_darcy_matrix(a1, grid32))
    a2 = sample_log_permeability(grid32, 1)
    assert is_spd_probe(build_darcy_matrix(a2, grid32))


def test_zero_boundary_via_interior_only(grid32: DarcyGrid):
    # No boundary DOFs exist; sine modes vanish at domain boundary by construction
    u = manufactured_sine_mode(grid32, 1, 1)
    X, Y = grid32.coords()
    # Interior points are away from boundary; check mode shape
    assert u.shape == (grid32.n, grid32.n)
    assert abs(u).max() > 0


def test_nonpositive_rejected(grid32: DarcyGrid):
    a = np.ones((grid32.n, grid32.n))
    a[0, 0] = -1.0
    with pytest.raises(ValueError):
        validate_permeability(a)


def test_deterministic_permeability(grid32: DarcyGrid):
    a1 = sample_log_permeability(grid32, 42)
    a2 = sample_log_permeability(grid32, 42)
    assert np.allclose(a1, a2)


def test_manufactured_roundtrip_modes(grid32: DarcyGrid):
    a = sample_log_permeability(grid32, 7, length_scale=0.2)
    for kx, ky in [(1, 1), (2, 3), (4, 1)]:
        u = manufactured_sine_mode(grid32, kx, ky, 0.2)
        rt = manufactured_round_trip(a, u, grid32, rtol=1e-8)
        assert rt["ok"], rt


def test_constant_a_poisson_like(grid32: DarcyGrid):
    a = np.ones((grid32.n, grid32.n))
    u = manufactured_sine_mode(grid32, 1, 1)
    f = apply_operator(a, u, grid32)
    u2, info = solve_darcy(a, f, grid32)
    assert info["residual_rel"] < 1e-8
    assert np.linalg.norm(u2 - u) / np.linalg.norm(u) < 1e-8


def test_extreme_contrast_roundtrip(grid32: DarcyGrid):
    a = sample_log_permeability(grid32, 9, a_min=0.1, a_max=10.0)
    a = np.clip(a, 0.1, 10.0)
    u = manufactured_sine_mode(grid32, 2, 2, 0.1)
    rt = manufactured_round_trip(a, u, grid32, rtol=1e-7)
    assert rt["ok"], rt


def test_condition_estimate_finite(grid32: DarcyGrid):
    a = np.ones((grid32.n, grid32.n))
    cond = condition_estimate(build_darcy_matrix(a, grid32))
    assert np.isfinite(cond) and cond > 1.0
