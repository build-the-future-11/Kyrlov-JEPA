"""Research-100 registry and lightweight mathematical sanity tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "20_run_research100_suite.py"
SPEC = importlib.util.spec_from_file_location("research100_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_research100_registry_is_complete():
    ids = [i for i, _ in mod.IDEAS]
    assert ids == list(range(1, 101))
    assert set(mod.CATEGORY) == set(ids)
    assert set(mod.MATURITY) == set(ids)


def test_sine_basis_is_orthonormal_in_euclidean_coordinates():
    grid = mod.GridSpec(n_interior=6)
    basis, _ = mod.sine_basis(grid, 3)
    gram = basis.T @ basis
    assert np.allclose(gram, np.eye(9), atol=1e-10, rtol=1e-10)


def test_magnetic_hamiltonian_is_hermitian():
    n = 5
    v = np.zeros((n, n))
    h = mod.magnetic_hamiltonian(n, v)
    assert np.max(np.abs(h - h.conj().T)) < 1e-12
