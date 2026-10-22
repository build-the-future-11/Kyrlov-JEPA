"""Evaluation metrics for ground-state prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


def _scaled_state_action(
    potential: np.ndarray,
    state: np.ndarray,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H after removing the arbitrary amplitude of a real state.

    These evaluation quantities are homogeneous of degree zero in the state.
    A fixed denominator floor changes that mathematical contract and can turn
    an underflowed or zero prediction into a perfect residual. Scaling before
    the Hamiltonian action also avoids overflow from a large state amplitude.
    """
    if np.iscomplexobj(state):
        raise ValueError("state must be real for this real-valued Hamiltonian evaluator")
    flat = np.asarray(state, dtype=np.float64).reshape(-1)
    if flat.size != grid.n_dof:
        raise ValueError(f"state size {flat.size} != grid degrees of freedom {grid.n_dof}")
    if not np.all(np.isfinite(flat)):
        raise ValueError("state must contain only finite values")
    scale = float(np.max(np.abs(flat)))
    if scale == 0.0:
        raise ValueError("state must be nonzero; a zero prediction has no Rayleigh quotient or relative residual")
    flat = flat / scale
    ham = build_hamiltonian(grid, potential)
    with np.errstate(over="ignore", invalid="ignore"):
        hflat = ham.matvec(flat)
    if not np.all(np.isfinite(hflat)):
        raise ValueError("Hamiltonian action is not representable with finite float64 values")
    return flat, hflat


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
    """||H ψ̂ - Ê ψ̂||₂ / ||ψ̂||₂, invariant to nonzero state amplitude.

    Undefined zero/nonfinite states and nonfinite energies are rejected, not
    assigned a zero residual. Norms are scaled before their square reductions.
    """
    energy = float(e_hat)
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    flat, hflat = _scaled_state_action(potential, psi_hat, grid)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = hflat - energy * flat
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual is not representable with finite float64 values")
    scale = float(np.max(np.abs(residual)))
    if scale == 0.0:
        return 0.0
    value = scale * float(np.linalg.norm(residual / scale) / np.linalg.norm(flat))
    if not np.isfinite(value):
        raise ValueError("residual norm is not representable with finite float64 values")
    return value


def rayleigh_quotient(
    potential: np.ndarray,
    psi: np.ndarray,
    grid: GridSpec,
) -> float:
    """E_R = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ for a finite nonzero real state.

    Scale the state and Hamiltonian action separately so a representable
    quotient does not require representable unnormalized squared amplitudes.
    """
    flat, hflat = _scaled_state_action(potential, psi, grid)
    scale = float(np.max(np.abs(hflat)))
    if scale == 0.0:
        return 0.0
    value = float(np.dot(flat, hflat / scale) / np.dot(flat, flat)) * scale
    if not np.isfinite(value):
        raise ValueError("Rayleigh quotient is not representable with finite float64 values")
    return value


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
