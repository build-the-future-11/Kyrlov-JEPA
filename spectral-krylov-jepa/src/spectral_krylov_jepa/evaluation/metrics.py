"""Evaluation metrics for ground-state prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


def fidelity_np(psi_hat: np.ndarray, psi: np.ndarray, grid: GridSpec) -> float:
    """Sign-invariant fidelity |⟨ψ̂, ψ⟩|² for normalized states."""
    area = cell_area(grid)
    a = np.asarray(psi_hat, dtype=np.float64).reshape(-1)
    b = np.asarray(psi, dtype=np.float64).reshape(-1)
    return float(np.abs(np.dot(a, b) * area) ** 2)


def relative_energy_error(e_hat: float, e_true: float, eps: float = 1e-6) -> float:
    return float(abs(e_hat - e_true) / (abs(e_true) + eps))


def schrodinger_residual(
    potential: np.ndarray,
    psi_hat: np.ndarray,
    e_hat: float,
    grid: GridSpec,
) -> float:
    """||H ψ̂ - Ê ψ̂||₂ / ||ψ̂||₂ in Euclidean vector norm."""
    ham = build_hamiltonian(grid, potential)
    flat = np.asarray(psi_hat, dtype=np.float64).reshape(-1)
    r = ham.matvec(flat) - e_hat * flat
    return float(np.linalg.norm(r) / max(np.linalg.norm(flat), 1e-15))


def rayleigh_quotient(
    potential: np.ndarray,
    psi: np.ndarray,
    grid: GridSpec,
) -> float:
    """E_R = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ using Euclidean inner product (consistent FD)."""
    ham = build_hamiltonian(grid, potential)
    flat = np.asarray(psi, dtype=np.float64).reshape(-1)
    hq = ham.matvec(flat)
    denom = float(np.dot(flat, flat))
    if denom < 1e-30:
        return float("nan")
    return float(np.dot(flat, hq) / denom)


def sign_aligned_l2(psi_hat: np.ndarray, psi: np.ndarray, grid: GridSpec) -> float:
    """Relative L2 after aligning global sign."""
    area = cell_area(grid)
    a = np.asarray(psi_hat, dtype=np.float64).reshape(-1)
    b = np.asarray(psi, dtype=np.float64).reshape(-1)
    if np.dot(a, b) < 0:
        a = -a
    num = np.sqrt(np.sum((a - b) ** 2) * area)
    den = np.sqrt(np.sum(b ** 2) * area) + 1e-15
    return float(num / den)


def evaluate_example(
    potential: np.ndarray,
    psi_true: np.ndarray,
    e_true: float,
    psi_hat: np.ndarray,
    e_hat: float,
    grid: GridSpec,
) -> dict[str, float]:
    """Full metric suite for one example.

    ``residual_rel`` uses the model energy head (protocol primary residual).
    ``residual_true_e`` isolates wavefunction quality at the true eigenvalue.
    ``residual_rayleigh`` uses E_R from ψ̂ (variational residual; small iff ψ̂
    is nearly an eigenvector of H).
    """
    e_rayleigh = rayleigh_quotient(potential, psi_hat, grid)
    return {
        "fidelity": fidelity_np(psi_hat, psi_true, grid),
        "infidelity": 1.0 - fidelity_np(psi_hat, psi_true, grid),
        "rel_energy_error": relative_energy_error(e_hat, e_true),
        "rel_rayleigh_energy_error": relative_energy_error(e_rayleigh, e_true),
        "rayleigh_energy": e_rayleigh,
        "residual_rel": schrodinger_residual(potential, psi_hat, e_hat, grid),
        "residual_true_e": schrodinger_residual(potential, psi_hat, e_true, grid),
        "residual_rayleigh": schrodinger_residual(potential, psi_hat, e_rayleigh, grid),
        "sign_aligned_rel_l2": sign_aligned_l2(psi_hat, psi_true, grid),
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k != "index"]
    out: dict[str, Any] = {"n": len(rows)}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
        out[f"{k}_values"] = vals.tolist()
    return out
