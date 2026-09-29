"""Physics-first hybrid spectral solvers for ground-state prediction.

The core idea is to keep an exact low-frequency Rayleigh--Ritz subspace and add
one or more potential-dependent proposal directions. The final eigenpair is
always obtained by an exact projected eigensolve, so learned components propose
a subspace rather than directly claiming a physical eigenstate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from spectral_krylov_jepa.physics.eigensolver import normalize_wavefunction
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


@dataclass(frozen=True)
class AdaptiveRitzResult:
    energy: float
    wavefunction: np.ndarray
    basis_dim: int
    proposal_rank: int
    projected_eigenvalues: np.ndarray


@lru_cache(maxsize=64)
def _cached_sine_basis(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= side <= n:
        raise ValueError(f"side must be in [1, {n}], got {side}")
    hx = (x_max - x_min) / (n + 1)
    hy = (y_max - y_min) / (n + 1)
    x = x_min + hx * np.arange(1, n + 1, dtype=np.float64)
    y = y_min + hy * np.arange(1, n + 1, dtype=np.float64)
    x = (x - x_min) / (x_max - x_min)
    y = (y - y_min) / (y_max - y_min)
    cols: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    for my in range(1, side + 1):
        sy = np.sin(np.pi * my * y)
        for mx in range(1, side + 1):
            q = np.outer(sy, np.sin(np.pi * mx * x)).reshape(-1)
            q /= max(float(np.linalg.norm(q)), 1e-15)
            cols.append(q)
            labels.append((mx, my))
    basis = np.stack(cols, axis=1)
    labels_arr = np.asarray(labels, dtype=np.int64)
    basis.setflags(write=False)
    labels_arr.setflags(write=False)
    return basis, labels_arr


def sine_basis(grid: GridSpec, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached Euclidean-orthonormal Dirichlet sine modes."""
    return _cached_sine_basis(
        grid.n_interior,
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
        int(side),
    )


@lru_cache(maxsize=64)
def _cached_box_energies(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    side: int,
) -> np.ndarray:
    _, labels = _cached_sine_basis(
        n, x_min, x_max, y_min, y_max, side
    )
    hx = (x_max - x_min) / (n + 1)
    hy = (y_max - y_min) / (n + 1)
    mx = labels[:, 0].astype(np.float64)
    my = labels[:, 1].astype(np.float64)
    energies = (
        2.0 / (hx * hx) * np.sin(np.pi * mx / (2.0 * (n + 1))) ** 2
        + 2.0 / (hy * hy) * np.sin(np.pi * my / (2.0 * (n + 1))) ** 2
    )
    energies.setflags(write=False)
    return energies


def box_energies(grid: GridSpec, side: int) -> np.ndarray:
    return _cached_box_energies(
        grid.n_interior,
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
        int(side),
    )


def spectral_coefficients(psi: np.ndarray, grid: GridSpec, side: int) -> np.ndarray:
    basis, _ = sine_basis(grid, side)
    q = np.asarray(psi, dtype=np.float64).reshape(-1).copy()
    q /= max(float(np.linalg.norm(q)), 1e-15)
    coeff = basis.T @ q
    if coeff[0] < 0:
        coeff = -coeff
    return coeff


def reconstruct_from_coefficients(coeff: np.ndarray, grid: GridSpec, side: int) -> np.ndarray:
    basis, _ = sine_basis(grid, side)
    c = np.asarray(coeff, dtype=np.float64).reshape(-1)
    if c.size != basis.shape[1]:
        raise ValueError(f"Expected {basis.shape[1]} coefficients, got {c.size}")
    return normalize_wavefunction((basis @ c).reshape(grid.ny, grid.nx), grid)


def low_mode_mask(side: int, low_side: int) -> np.ndarray:
    if not 1 <= low_side <= side:
        raise ValueError("low_side must be between 1 and side")
    labels = [(mx, my) for my in range(1, side + 1) for mx in range(1, side + 1)]
    return np.asarray(
        [mx <= low_side and my <= low_side for mx, my in labels],
        dtype=bool,
    )




def first_order_perturbation_fast(
    potential: np.ndarray,
    grid: GridSpec,
    side: int,
) -> tuple[np.ndarray, float]:
    """First-order box perturbation using only ground-to-mode couplings.

    Unlike the second-order routine, this does not construct the full projected
    potential matrix. It needs one weighted projection per basis mode.
    """
    basis, _ = sine_basis(grid, side)
    e = box_energies(grid, side)
    v = np.asarray(potential, dtype=np.float64).reshape(-1)
    weighted_ground = v * basis[:, 0]
    coupling = basis.T @ weighted_ground

    coeff = np.zeros(basis.shape[1], dtype=np.float64)
    coeff[0] = 1.0
    e0 = float(e[0])
    for n in range(1, len(coeff)):
        denom = e0 - float(e[n])
        if abs(denom) >= 1e-14:
            coeff[n] = float(coupling[n]) / denom
    energy = e0 + float(coupling[0])
    return coeff, float(energy)


def first_second_order_perturbation(
    potential: np.ndarray,
    grid: GridSpec,
    side: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return first/second-order coefficient vectors and E1/E2."""
    basis, _ = sine_basis(grid, side)
    e = box_energies(grid, side)
    v = np.asarray(potential, dtype=np.float64).reshape(-1)
    vmat = basis.T @ (v[:, None] * basis)

    c1 = np.zeros(basis.shape[1], dtype=np.float64)
    c2 = np.zeros_like(c1)
    c1[0] = 1.0
    c2[0] = 1.0
    e0 = float(e[0])
    v00 = float(vmat[0, 0])

    for n in range(1, len(c1)):
        dn = e0 - float(e[n])
        if abs(dn) < 1e-14:
            continue
        c1[n] = float(vmat[n, 0]) / dn
        acc = 0.0
        for m in range(1, len(c1)):
            dm = e0 - float(e[m])
            if abs(dm) < 1e-14:
                continue
            acc += float(vmat[n, m] * vmat[m, 0]) / (dn * dm)
        acc -= v00 * float(vmat[n, 0]) / (dn * dn)
        c2[n] = c1[n] + acc

    e1 = e0 + v00
    e2 = e1
    for m in range(1, len(c1)):
        dm = e0 - float(e[m])
        if abs(dm) >= 1e-14:
            e2 += float(vmat[0, m] * vmat[m, 0]) / dm
    return c1, c2, float(e1), float(e2)


def _projected_ritz_orthonormal(
    potential: np.ndarray,
    grid: GridSpec,
    q: np.ndarray,
) -> AdaptiveRitzResult:
    """Solve a projected eigenproblem for an already orthonormal basis."""
    ham = build_hamiltonian(grid, potential)
    hp = q.T @ (ham.matrix @ q)
    hp = (hp + hp.T) / 2.0
    vals, vecs = np.linalg.eigh(hp)
    wave = q @ vecs[:, 0]
    psi = normalize_wavefunction(wave.reshape(grid.ny, grid.nx), grid)
    return AdaptiveRitzResult(
        energy=float(vals[0]),
        wavefunction=psi,
        basis_dim=int(q.shape[1]),
        proposal_rank=0,
        projected_eigenvalues=vals.astype(float),
    )


def projected_ritz(
    potential: np.ndarray,
    grid: GridSpec,
    basis: np.ndarray,
    *,
    assume_orthonormal: bool = False,
) -> AdaptiveRitzResult:
    """Solve the exact Rayleigh--Ritz problem in an arbitrary basis."""
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != grid.n_dof:
        raise ValueError(f"basis must have shape ({grid.n_dof}, K)")
    q = b if assume_orthonormal else np.linalg.qr(b)[0]
    return _projected_ritz_orthonormal(potential, grid, q)


def adaptive_ritz(
    potential: np.ndarray,
    grid: GridSpec,
    *,
    low_side: int = 3,
    proposal_vectors: Iterable[np.ndarray] = (),
    orthogonal_tol: float = 1e-10,
) -> AdaptiveRitzResult:
    """Augment a fixed low-mode sine basis with potential-dependent directions."""
    low_basis, _ = sine_basis(grid, low_side)
    cols = [low_basis[:, j].copy() for j in range(low_basis.shape[1])]
    proposal_rank = 0

    for proposal in proposal_vectors:
        v = np.asarray(proposal, dtype=np.float64).reshape(-1).copy()
        if v.size != grid.n_dof:
            raise ValueError(f"proposal has {v.size} entries; expected {grid.n_dof}")
        for q in cols:
            v -= float(np.dot(q, v)) * q
        nrm = float(np.linalg.norm(v))
        if nrm <= orthogonal_tol:
            continue
        v /= nrm
        cols.append(v)
        proposal_rank += 1

    result = projected_ritz(
        potential,
        grid,
        np.stack(cols, axis=1),
        assume_orthonormal=True,
    )
    return AdaptiveRitzResult(
        energy=result.energy,
        wavefunction=result.wavefunction,
        basis_dim=result.basis_dim,
        proposal_rank=proposal_rank,
        projected_eigenvalues=result.projected_eigenvalues,
    )


def coefficient_direction(
    coeff: np.ndarray,
    grid: GridSpec,
    *,
    side: int,
    low_side: int,
) -> np.ndarray:
    c = np.asarray(coeff, dtype=np.float64).reshape(-1).copy()
    mask = low_mode_mask(side, low_side)
    if c.size != mask.size:
        raise ValueError(f"Expected {mask.size} coefficients, got {c.size}")
    c[mask] = 0.0
    basis, _ = sine_basis(grid, side)
    return basis @ c


def physics_features(
    potential: np.ndarray,
    grid: GridSpec,
    *,
    side: int = 7,
) -> np.ndarray:
    """Compact operator-derived features with no exact eigenstate labels."""
    basis, _ = sine_basis(grid, side)
    e = box_energies(grid, side)
    v = np.asarray(potential, dtype=np.float64)
    vf = v.reshape(-1)
    vmat = basis.T @ (vf[:, None] * basis)
    gaps = e[0] - e
    safe = np.where(np.abs(gaps) < 1e-12, 1.0, gaps)

    coupling = vmat[:, 0] / safe
    coupling[0] = 0.0
    diagonal = np.diag(vmat) - vmat[0, 0]
    gy, gx = np.gradient(v)
    stats = np.asarray(
        [
            float(v.mean()),
            float(v.std()),
            float(v.min()),
            float(v.max()),
            float(np.mean(gx * gx + gy * gy)),
            float(np.linalg.norm(vf)),
        ],
        dtype=np.float64,
    )
    _, c2, _, e2 = first_second_order_perturbation(v, grid, side)
    return np.concatenate([coupling, diagonal, c2, np.asarray([e2]), stats])


def embed_low_ritz_coefficients(
    potential: np.ndarray,
    grid: GridSpec,
    *,
    side: int,
    low_side: int,
) -> tuple[np.ndarray, AdaptiveRitzResult]:
    low_basis, _ = sine_basis(grid, low_side)
    low = projected_ritz(potential, grid, low_basis)
    coeff = spectral_coefficients(low.wavefunction, grid, side)
    mask = low_mode_mask(side, low_side)
    coeff[~mask] = 0.0
    return coeff, low


def high_mode_target(
    psi_true: np.ndarray,
    grid: GridSpec,
    *,
    side: int,
    low_side: int,
) -> np.ndarray:
    coeff = spectral_coefficients(psi_true, grid, side)
    coeff[low_mode_mask(side, low_side)] = 0.0
    return coeff


def perturbation_high_mode_direction(
    potential: np.ndarray,
    grid: GridSpec,
    *,
    side: int,
    low_side: int,
    order: int = 2,
) -> np.ndarray:
    if order == 1:
        coeff, _ = first_order_perturbation_fast(potential, grid, side)
    else:
        _, coeff, _, _ = first_second_order_perturbation(potential, grid, side)
    return coefficient_direction(coeff, grid, side=side, low_side=low_side)


def variational_monotonicity_gap(
    small: AdaptiveRitzResult,
    large: AdaptiveRitzResult,
) -> float:
    return float(small.energy - large.energy)
