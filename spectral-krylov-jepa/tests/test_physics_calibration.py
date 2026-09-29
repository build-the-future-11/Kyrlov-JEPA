"""Tests for physics-aware calibration losses and spectral decoder."""

import numpy as np
import torch

from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_box_potential
from spectral_krylov_jepa.training.losses import (
    apply_hamiltonian_torch,
    hamiltonian_residual_loss,
    hamiltonian_residual_per_example,
    rayleigh_consistency_loss,
)


def test_torch_hamiltonian_matches_sparse_reference():
    g = GridSpec(n_interior=8)
    rng = np.random.default_rng(7)
    v = rng.normal(size=(g.ny, g.nx))
    psi = rng.normal(size=(g.ny, g.nx))

    ham = build_hamiltonian(g, v)
    expected = ham.matvec(psi.reshape(-1)).reshape(g.ny, g.nx)

    got = apply_hamiltonian_torch(
        torch.from_numpy(v).unsqueeze(0).double(),
        torch.from_numpy(psi).unsqueeze(0).double(),
        grid_spacing=g.h,
    )[0].numpy()
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_true_eigenpair_has_tiny_torch_residual():
    g = GridSpec(n_interior=10)
    v, _ = generate_box_potential(g)
    gs = solve_ground_state(build_hamiltonian(g, v))

    residual = hamiltonian_residual_per_example(
        torch.from_numpy(v).unsqueeze(0).double(),
        torch.from_numpy(gs.wavefunction).unsqueeze(0).double(),
        torch.tensor([gs.energy], dtype=torch.float64),
        grid_spacing=g.h,
        scaled=False,
    )
    assert float(residual.item()) < 1e-5


def test_physics_losses_backpropagate():
    g = GridSpec(n_interior=8)
    v = torch.randn(2, 8, 8, dtype=torch.float64)
    psi = torch.randn(2, 8, 8, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(2, dtype=torch.float64, requires_grad=True)

    loss = hamiltonian_residual_loss(
        v,
        psi,
        energy,
        grid_spacing=g.h,
        mode="scaled",
    )
    loss = loss + 0.1 * rayleigh_consistency_loss(
        v,
        psi,
        energy,
        grid_spacing=g.h,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert psi.grad is not None and torch.isfinite(psi.grad).all()
    assert energy.grad is not None and torch.isfinite(energy.grad).all()


def test_sine_decoder_downstream_is_normalized():
    g = GridSpec(n_interior=16)
    model = DownstreamGroundStateModel(
        img_size=16,
        size="smoke",
        cell_area=cell_area(g),
        decoder_type="sine",
        sine_modes=25,
    )
    out = model(torch.randn(3, 16, 16))
    assert out.psi.shape == (3, 16, 16)
    assert torch.isfinite(out.psi).all()
    norms = torch.sum(out.psi * out.psi, dim=(-2, -1)) * cell_area(g)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
