"""Tests for physics-aware calibration losses and spectral decoder."""

import numpy as np
import torch

from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.models.krylov_jepa import KrylovJEPA
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


def test_projected_ritz_target_matches_free_box_ground_energy():
    g = GridSpec(n_interior=16)
    v, _ = generate_box_potential(g)
    gs = solve_ground_state(build_hamiltonian(g, v))
    model = KrylovJEPA(
        img_size=16,
        size="smoke",
        context_steps=2,
        lambda_projected_ritz=1.0,
        projected_modes=9,
    )
    with torch.no_grad():
        e_target, coeff = model._projected_ritz_targets(
            torch.from_numpy(v).unsqueeze(0).float()
        )
    assert abs(float(e_target.item()) - gs.energy) / abs(gs.energy) < 1e-5
    assert coeff.shape == (1, 9)
    assert torch.isfinite(coeff).all()


def test_projected_ritz_auxiliary_loss_reaches_potential_encoder():
    model = KrylovJEPA(
        img_size=16,
        size="smoke",
        context_steps=2,
        lambda_projected_ritz=1.0,
        projected_modes=9,
    )
    v = torch.randn(2, 16, 16)
    q_ctx = torch.randn(2, 2, 256)
    q_ctx = q_ctx / q_ctx.norm(dim=-1, keepdim=True)
    q_tgt = torch.randn(2, 256)
    q_tgt = q_tgt / q_tgt.norm(dim=-1, keepdim=True)
    alpha = torch.randn(2, 3)
    beta = torch.rand(2, 3) + 0.1
    out = model(v, q_ctx, q_tgt, alpha=alpha, beta=beta)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.loss_projected_ritz)
    assert float(out.loss_projected_ritz.item()) > 0.0
    out.loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.potential_enc.parameters()
    )
