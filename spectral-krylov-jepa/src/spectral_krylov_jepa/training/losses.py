"""Training losses for JEPA and downstream fine-tuning."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def jepa_mse(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return torch.mean((z_pred - z_target.detach()) ** 2)


def wavefunction_fidelity(psi_hat: torch.Tensor, psi: torch.Tensor, cell_area: float) -> torch.Tensor:
    """Batch fidelity with the discrete L2 inner product."""
    a = psi_hat.reshape(psi_hat.shape[0], -1)
    b = psi.reshape(psi.shape[0], -1)
    inner = torch.sum(a * b, dim=-1) * cell_area
    return inner.abs().pow(2)


def sign_invariant_psi_loss(
    psi_hat: torch.Tensor,
    psi: torch.Tensor,
    cell_area: float,
) -> torch.Tensor:
    """1 - fidelity."""
    f = wavefunction_fidelity(psi_hat, psi, cell_area)
    return torch.mean(1.0 - f)


def sign_aligned_mse(
    psi_hat: torch.Tensor,
    psi: torch.Tensor,
    cell_area: float,
) -> torch.Tensor:
    """Sign-aligned spatial MSE."""
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
    """MSE in z-score space."""
    std = energy_std if not torch.is_tensor(energy_std) else energy_std.clamp_min(1e-6)
    mean = energy_mean
    z_hat = (e_hat - mean) / std
    z_true = (e_true - mean) / std
    return torch.mean((z_hat - z_true) ** 2)


def apply_hamiltonian_torch(
    potential: torch.Tensor,
    psi: torch.Tensor,
    *,
    grid_spacing: float | None = None,
) -> torch.Tensor:
    """Apply H = -0.5 Laplacian + V to interior-grid states.

    Dirichlet boundary values are fixed to zero. This mirrors the sparse
    finite-difference Hamiltonian while remaining differentiable in psi.
    """
    if psi.ndim != 3:
        raise ValueError(f"psi must have shape (B,H,W), got {tuple(psi.shape)}")
    if potential.ndim == 2:
        potential = potential.unsqueeze(0)
    if potential.ndim != 3:
        raise ValueError(
            f"potential must have shape (B,H,W) or (H,W), got {tuple(potential.shape)}"
        )
    if potential.shape[-2:] != psi.shape[-2:]:
        raise ValueError(
            f"potential spatial shape {tuple(potential.shape[-2:])} != "
            f"psi spatial shape {tuple(psi.shape[-2:])}"
        )
    if potential.shape[0] not in (1, psi.shape[0]):
        raise ValueError(
            f"potential batch {potential.shape[0]} is incompatible with psi batch {psi.shape[0]}"
        )

    height, width = psi.shape[-2:]
    if grid_spacing is None:
        if height != width:
            raise ValueError("grid_spacing is required for non-square grids")
        h = 1.0 / float(width + 1)
    else:
        h = float(grid_spacing)
    if not math.isfinite(h) or h <= 0:
        raise ValueError(f"grid_spacing must be positive and finite, got {h}")

    x = psi.unsqueeze(1)
    padded = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
    center = padded[:, :, 1:-1, 1:-1]
    lap = (
        padded[:, :, 1:-1, 2:]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, :-2, 1:-1]
        - 4.0 * center
    ) / (h * h)
    kinetic = -0.5 * lap.squeeze(1)
    return kinetic + potential.to(dtype=psi.dtype, device=psi.device) * psi


def rayleigh_energy_torch(
    potential: torch.Tensor,
    psi: torch.Tensor,
    *,
    grid_spacing: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Rayleigh quotient of each predicted state."""
    hpsi = apply_hamiltonian_torch(potential, psi, grid_spacing=grid_spacing)
    flat_psi = psi.reshape(psi.shape[0], -1)
    flat_hpsi = hpsi.reshape(hpsi.shape[0], -1)
    denom = torch.sum(flat_psi * flat_psi, dim=-1).clamp_min(eps)
    return torch.sum(flat_psi * flat_hpsi, dim=-1) / denom


def hamiltonian_residual_per_example(
    potential: torch.Tensor,
    psi: torch.Tensor,
    energy: torch.Tensor,
    *,
    grid_spacing: float | None = None,
    scaled: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Hamiltonian residual for each example.

    scaled=False returns ||H psi - E psi|| / ||psi||.
    scaled=True normalizes by ||H psi|| + ||E psi|| for stable optimization.
    """
    hpsi = apply_hamiltonian_torch(potential, psi, grid_spacing=grid_spacing)
    epsi = energy.reshape(-1, 1, 1) * psi
    residual = hpsi - epsi
    num = torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=-1)
    if scaled:
        denom = (
            torch.linalg.vector_norm(hpsi.reshape(hpsi.shape[0], -1), dim=-1)
            + torch.linalg.vector_norm(epsi.reshape(epsi.shape[0], -1), dim=-1)
        ).clamp_min(eps)
    else:
        denom = torch.linalg.vector_norm(psi.reshape(psi.shape[0], -1), dim=-1).clamp_min(eps)
    return num / denom


def hamiltonian_residual_loss(
    potential: torch.Tensor,
    psi: torch.Tensor,
    energy: torch.Tensor,
    *,
    grid_spacing: float | None = None,
    mode: str = "scaled",
) -> torch.Tensor:
    """Mean squared Hamiltonian residual for differentiable training."""
    if mode not in {"scaled", "raw"}:
        raise ValueError(f"Unknown residual mode {mode!r}; expected scaled or raw")
    values = hamiltonian_residual_per_example(
        potential,
        psi,
        energy,
        grid_spacing=grid_spacing,
        scaled=mode == "scaled",
    )
    return torch.mean(values * values)


def rayleigh_consistency_loss(
    potential: torch.Tensor,
    psi: torch.Tensor,
    energy: torch.Tensor,
    *,
    grid_spacing: float | None = None,
) -> torch.Tensor:
    """Keep the energy head consistent with the state Rayleigh quotient."""
    e_rayleigh = rayleigh_energy_torch(
        potential,
        psi,
        grid_spacing=grid_spacing,
    )
    scale = e_rayleigh.detach().abs().clamp_min(1.0)
    return torch.mean(((energy - e_rayleigh) / scale) ** 2)


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
    *,
    potential: torch.Tensor | None = None,
    lambda_residual: float = 0.0,
    lambda_rayleigh: float = 0.0,
    residual_mode: str = "scaled",
    grid_spacing: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined supervised and physics-aware downstream objective.

    Legacy behavior is preserved when both physics weights are zero.
    """
    l_psi = sign_invariant_psi_loss(psi_hat, psi_true, cell_area)
    l_psi_mse = sign_aligned_mse(psi_hat, psi_true, cell_area)
    if energy_mean is not None and energy_std is not None:
        l_e = standardized_energy_mse(e_hat, e_true, energy_mean, energy_std)
    else:
        l_e = normalized_energy_mse(e_hat, e_true)

    loss = lambda_psi * l_psi + lambda_psi_mse * l_psi_mse + lambda_e * l_e
    l_residual = torch.zeros((), dtype=loss.dtype, device=loss.device)
    l_rayleigh = torch.zeros((), dtype=loss.dtype, device=loss.device)
    raw_residual = torch.full((), float("nan"), dtype=loss.dtype, device=loss.device)

    if lambda_residual != 0.0 or lambda_rayleigh != 0.0:
        if potential is None:
            raise ValueError("potential is required when physics-aware loss terms are enabled")
        raw_values = hamiltonian_residual_per_example(
            potential,
            psi_hat,
            e_hat,
            grid_spacing=grid_spacing,
            scaled=False,
        )
        raw_residual = raw_values.mean()
        if lambda_residual != 0.0:
            l_residual = hamiltonian_residual_loss(
                potential,
                psi_hat,
                e_hat,
                grid_spacing=grid_spacing,
                mode=residual_mode,
            )
            loss = loss + lambda_residual * l_residual
        if lambda_rayleigh != 0.0:
            l_rayleigh = rayleigh_consistency_loss(
                potential,
                psi_hat,
                e_hat,
                grid_spacing=grid_spacing,
            )
            loss = loss + lambda_rayleigh * l_rayleigh

    with torch.no_grad():
        fidelity = wavefunction_fidelity(psi_hat, psi_true, cell_area).mean().item()
    return loss, {
        "loss_psi": float(l_psi.item()),
        "loss_psi_mse": float(l_psi_mse.item()),
        "loss_e": float(l_e.item()),
        "loss_residual": float(l_residual.item()),
        "loss_rayleigh": float(l_rayleigh.item()),
        "residual_raw": float(raw_residual.item()),
        "fidelity": fidelity,
    }
