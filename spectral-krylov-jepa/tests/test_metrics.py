"""Metric tests."""

import numpy as np

from spectral_krylov_jepa.evaluation.metrics import fidelity_np, schrodinger_residual
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_box_potential


def test_fidelity_sign_invariance():
    g = GridSpec(n_interior=8)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    f1 = fidelity_np(gs.wavefunction, gs.wavefunction, g)
    f2 = fidelity_np(-gs.wavefunction, gs.wavefunction, g)
    assert abs(f1 - 1.0) < 1e-8
    assert abs(f2 - 1.0) < 1e-8


def test_true_eigenpair_small_residual():
    g = GridSpec(n_interior=12)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    r = schrodinger_residual(v, gs.wavefunction, gs.energy, g)
    assert r < 1e-5


def test_rayleigh_matches_eigenvalue():
    from spectral_krylov_jepa.evaluation.metrics import rayleigh_quotient

    g = GridSpec(n_interior=12)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    e_r = rayleigh_quotient(v, gs.wavefunction, g)
    assert abs(e_r - gs.energy) / abs(gs.energy) < 1e-6
    # Residual at Rayleigh energy must be tiny for a true eigenpair
    assert schrodinger_residual(v, gs.wavefunction, e_r, g) < 1e-5


def test_evaluate_example_reports_dual_residuals():
    from spectral_krylov_jepa.evaluation.metrics import evaluate_example

    g = GridSpec(n_interior=8)
    v, _ = generate_box_potential(g)
    ham = build_hamiltonian(g, v)
    gs = solve_ground_state(ham)
    # Deliberately wrong energy head value
    m = evaluate_example(v, gs.wavefunction, gs.energy, gs.wavefunction, gs.energy * 1.5, g)
    assert m["residual_true_e"] < 1e-5
    assert m["residual_rel"] > m["residual_true_e"]
    assert m["residual_rayleigh"] < 1e-5
    assert m["fidelity"] > 0.999
