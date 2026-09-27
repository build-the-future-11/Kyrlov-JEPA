"""Training losses for JEPA and downstream fine-tuning."""

from __future__ import annotations

import torch


def jepa_mse(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return torch.mean((z_pred - z_target.detach()) ** 2)


def wavefunction_fidelity(psi_hat: torch.Tensor, psi: torch.Tensor, cell_area: float) -> torch.Tensor:
    """Batch fidelity F = |⟨ψ̂, ψ⟩|² with discrete L2 inner product."""
    a = psi_hat.reshape(psi_hat.shape[0], -1)
    b = psi.reshape(psi.shape[0], -1)
    inner = torch.sum(a * b, dim=-1) * cell_area
    return inner.abs().pow(2)


def sign_invariant_psi_loss(
    psi_hat: torch.Tensor,
    psi: torch.Tensor,
    cell_area: float,
) -> torch.Tensor:
    """1 - fidelity (primary wavefunction training loss)."""
    f = wavefunction_fidelity(psi_hat, psi, cell_area)
    return torch.mean(1.0 - f)


def sign_aligned_mse(
    psi_hat: torch.Tensor,
    psi: torch.Tensor,
    cell_area: float,
) -> torch.Tensor:
    """Sign-aligned spatial MSE (auxiliary sharpness on ψ)."""
    a = psi_hat.reshape(psi_hat.shape[0], -1)
    b = psi.reshape(psi.shape[0], -1)
    dots = torch.sum(a * b, dim=-1, keepdim=True)
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    a = a * signs
    return torch.mean(torch.sum((a - b) ** 2, dim=-1) * cell_area)


def normalized_energy_mse(
    e_hat: torch.Tensor,
    e_true: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean squared relative energy error in physical units."""
    rel = (e_hat - e_true) / (e_true.abs() + eps)
    return torch.mean(rel ** 2)


def standardized_energy_mse(
    e_hat: torch.Tensor,
    e_true: torch.Tensor,
    energy_mean: torch.Tensor | float,
    energy_std: torch.Tensor | float,
) -> torch.Tensor:
    """MSE in z-score space (stabilizes multi-scale energies)."""
    std = energy_std if not torch.is_tensor(energy_std) else energy_std.clamp_min(1e-6)
    mean = energy_mean
    z_hat = (e_hat - mean) / std
    z_true = (e_true - mean) / std
    return torch.mean((z_hat - z_true) ** 2)


def downstream_loss(
    e_hat: torch.Tensor,
    e_true: torch.Tensor,
    psi_hat: torch.Tensor,
    psi_true: torch.Tensor,
    cell_area: float,
    lambda_psi: float = 1.0,
    lambda_e: float = 1.0,
    lambda_psi_mse: float = 0.1,
    energy_mean: float | torch.Tensor | None = None,
    energy_std: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined downstream objective.

    L = λ_ψ (1 - F) + λ_ψmse sign-aligned MSE + λ_E energy_loss

    Energy loss uses z-score MSE when stats are provided, else relative MSE.
    Defaults (a priori, not tuned on test): λ_ψ=1.0, λ_E=1.0, λ_ψmse=0.1.
    """
    l_psi = sign_invariant_psi_loss(psi_hat, psi_true, cell_area)
    l_psi_mse = sign_aligned_mse(psi_hat, psi_true, cell_area)
    if energy_mean is not None and energy_std is not None:
        l_e = standardized_energy_mse(e_hat, e_true, energy_mean, energy_std)
    else:
        l_e = normalized_energy_mse(e_hat, e_true)
    loss = lambda_psi * l_psi + lambda_psi_mse * l_psi_mse + lambda_e * l_e
    with torch.no_grad():
        f = wavefunction_fidelity(psi_hat, psi_true, cell_area).mean().item()
    return loss, {
        "loss_psi": float(l_psi.item()),
        "loss_psi_mse": float(l_psi_mse.item()),
        "loss_e": float(l_e.item()),
        "fidelity": f,
    }
