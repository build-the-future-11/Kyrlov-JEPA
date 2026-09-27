"""Lanczos tests."""

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.lanczos import run_lanczos, validate_lanczos
from spectral_krylov_jepa.physics.potentials import generate_potential


def test_lanczos_orthogonality_and_recurrence():
    g = GridSpec(n_interior=16)
    v, _ = generate_potential("id_gaussian_mixture", 99, grid=g)
    ham = build_hamiltonian(g, v)
    lanc = run_lanczos(ham, depth=3, q0_seed=1, reorthogonalize=True)
    val = validate_lanczos(ham, lanc)
    assert val["ok"], val
    assert val["orthogonality_error"] < 1e-6
    assert val["recurrence_residual"] < 1e-5


def test_lanczos_reproducible():
    g = GridSpec(n_interior=12)
    v, _ = generate_potential("id_gaussian_mixture", 1, grid=g)
    ham = build_hamiltonian(g, v)
    a = run_lanczos(ham, depth=2, q0_seed=42)
    b = run_lanczos(ham, depth=2, q0_seed=42)
    assert (a.q == b.q).all()
    assert (a.alpha == b.alpha).all()
