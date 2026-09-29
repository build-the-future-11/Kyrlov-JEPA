from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "benchmarks" / "confirmatory_20260929"
sys.path.insert(0, str(PKG))
from benchmark_core import Grid, exact_ground_state, fixed_ritz, metrics, perturbation_ritz_hybrid, pt2


def test_zero_potential_methods_are_finite_and_deterministic():
    grid = Grid(n=6)
    v = np.zeros((6, 6), dtype=np.float64)
    e0, psi0 = exact_ground_state(v, grid)
    a = [fixed_ritz(v, grid, 2), pt2(v, grid, 2), perturbation_ritz_hybrid(v, grid, perturb_side=2, krylov_depth=2)]
    b = [fixed_ritz(v, grid, 2), pt2(v, grid, 2), perturbation_ritz_hybrid(v, grid, perturb_side=2, krylov_depth=2)]
    for x, y in zip(a, b):
        mx = metrics(v, x, e0, psi0, grid)
        assert np.isfinite(mx["fidelity"])
        assert 0.0 <= mx["fidelity"] <= 1.0 + 1e-10
        assert np.isclose(x.energy, y.energy, rtol=0, atol=1e-12)
        assert np.allclose(np.abs(x.psi), np.abs(y.psi), rtol=0, atol=1e-12)


def test_compute_accounting_is_method_specific():
    grid = Grid(n=6); v = np.zeros((6, 6), dtype=np.float64)
    ritz = fixed_ritz(v, grid, 2); perturb = pt2(v, grid, 2)
    hybrid = perturbation_ritz_hybrid(v, grid, perturb_side=2, krylov_depth=2)
    assert ritz.accounting.query_full_h_matvecs == 4
    assert perturb.accounting.query_full_h_matvecs == 0
    assert perturb.accounting.reusable_setup_h_matvecs == 4
    assert hybrid.accounting.query_full_h_matvecs == 4
    assert hybrid.accounting.lanczos_depth == 2
