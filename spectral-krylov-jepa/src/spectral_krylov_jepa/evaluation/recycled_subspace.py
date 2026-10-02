"""Budget-accounted recycled Ritz subspace control.

This module implements only the pre-registered classical control machinery.
It deliberately does not choose a refresh threshold and does not perform a
fallback eigensolve after refresh: both are protocol-level choices that must be
frozen before result-bearing evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import Hamiltonian, build_hamiltonian


@dataclass(frozen=True)
class RecycleBudget:
    """Dominant-operation ledger for one recycled-subspace query."""

    basis_rank: int
    operator_applications: int
    reduced_dimension: int
    refresh_requested: bool


@dataclass(frozen=True)
class RecycledRitzResult:
    """Rayleigh--Ritz result and fail-closed refresh decision."""

    energy: float
    wavefunction: np.ndarray
    residual_norm: float
    residual_relative_to_hx: float
    refresh_requested: bool
    budget: RecycleBudget


def build_recycle_space(
    vectors: Iterable[np.ndarray],
    *,
    n_dof: int,
    rank: int,
    orthogonal_tol: float = 1e-12,
) -> np.ndarray:
    """Build an ordered Euclidean-orthonormal recycle space.

    Vectors are consumed in the caller-supplied order. Linearly dependent or
    near-zero directions are skipped. The function fails closed if the legal
    prior information cannot supply the requested fixed rank.
    """
    if n_dof <= 0:
        raise ValueError("n_dof must be positive")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not np.isfinite(orthogonal_tol) or orthogonal_tol <= 0:
        raise ValueError("orthogonal_tol must be finite and positive")

    basis: list[np.ndarray] = []
    for vector in vectors:
        v = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
        if v.size != n_dof:
            raise ValueError(f"recycle vector has {v.size} entries; expected {n_dof}")
        if not np.all(np.isfinite(v)):
            raise ValueError("recycle vector contains non-finite values")

        for q in basis:
            v -= q * float(q @ v)
        for q in basis:
            v -= q * float(q @ v)

        norm = float(np.linalg.norm(v))
        if norm <= orthogonal_tol:
            continue
        basis.append(v / norm)
        if len(basis) == rank:
            break

    if len(basis) != rank:
        raise ValueError(
            f"requested recycle rank {rank}, but only {len(basis)} independent "
            "legal directions were available"
        )

    q = np.column_stack(basis)
    gram_error = float(np.linalg.norm(q.T @ q - np.eye(rank), ord=np.inf))
    if gram_error > max(1e-10, 100.0 * orthogonal_tol):
        raise RuntimeError(f"recycle-space orthogonality check failed: {gram_error:.3e}")
    return q


def solve_recycled_ritz(
    ham: Hamiltonian,
    recycle_basis: np.ndarray,
    *,
    refresh_threshold: float,
    denominator_eps: float = 1e-15,
) -> RecycledRitzResult:
    """Solve the reduced problem after recomputing the new operator action.

    refresh_threshold is intentionally required. It must be frozen outside
    this function using development-only information. The normalized residual
    is ||Hx - theta x|| / max(||Hx||, denominator_eps).

    The budget counts one full Hamiltonian application for every recycle-basis
    vector. The residual uses the already-computed H @ U product, so it does
    not require another full Hamiltonian application.
    """
    if not np.isfinite(refresh_threshold) or refresh_threshold < 0:
        raise ValueError("refresh_threshold must be finite and non-negative")
    if not np.isfinite(denominator_eps) or denominator_eps <= 0:
        raise ValueError("denominator_eps must be finite and positive")

    q = np.asarray(recycle_basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != ham.n_dof or q.shape[1] <= 0:
        raise ValueError(
            f"recycle_basis must have shape ({ham.n_dof}, rank>0); got {q.shape}"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("recycle_basis contains non-finite values")

    rank = int(q.shape[1])
    gram_error = float(np.linalg.norm(q.T @ q - np.eye(rank), ord=np.inf))
    if gram_error > 1e-8:
        raise ValueError(
            "recycle_basis must already be Euclidean-orthonormal; "
            f"Gram error={gram_error:.3e}"
        )

    # Critical fairness rule: always recompute H_new @ U. No projected matrix
    # from the previous Hamiltonian is accepted by this API.
    h_basis = np.column_stack([ham.matvec(q[:, j]) for j in range(rank)])
    projected = q.T @ h_basis
    projected = (projected + projected.T) / 2.0
    values, vectors = np.linalg.eigh(projected)

    coeff = vectors[:, 0]
    x = q @ coeff
    x_norm = float(np.linalg.norm(x))
    if x_norm <= denominator_eps:
        raise RuntimeError("projected Ritz vector is numerically zero")
    x /= x_norm

    # Reuse H @ U for exact Hx; no extra full matvec is hidden in the residual.
    hx = (h_basis @ coeff) / x_norm
    energy = float(values[0])
    residual = hx - energy * x
    residual_norm = float(np.linalg.norm(residual))
    residual_relative_to_hx = residual_norm / max(float(np.linalg.norm(hx)), denominator_eps)
    refresh_requested = bool(residual_relative_to_hx > refresh_threshold)

    area = ham.grid.hx * ham.grid.hy
    physical_norm = np.sqrt(float(np.sum(x * x)) * area)
    if physical_norm <= denominator_eps:
        raise RuntimeError("cannot physically normalize recycled Ritz vector")
    psi = (x / physical_norm).reshape(ham.grid.ny, ham.grid.nx)

    flat = psi.reshape(-1)
    idx = int(np.argmax(np.abs(flat)))
    if flat[idx] < 0:
        psi = -psi

    budget = RecycleBudget(
        basis_rank=rank,
        operator_applications=rank,
        reduced_dimension=rank,
        refresh_requested=refresh_requested,
    )
    return RecycledRitzResult(
        energy=energy,
        wavefunction=psi,
        residual_norm=residual_norm,
        residual_relative_to_hx=residual_relative_to_hx,
        refresh_requested=refresh_requested,
        budget=budget,
    )


def solve_recycled_ritz_from_potential(
    potential: np.ndarray,
    grid: GridSpec,
    recycle_basis: np.ndarray,
    *,
    refresh_threshold: float,
    denominator_eps: float = 1e-15,
) -> RecycledRitzResult:
    """Build the new Hamiltonian and run the budget-accounted recycled solve."""
    ham = build_hamiltonian(grid, potential)
    return solve_recycled_ritz(
        ham,
        recycle_basis,
        refresh_threshold=refresh_threshold,
        denominator_eps=denominator_eps,
    )
