"""Sparse symmetric eigensolver for ground states of H."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import Hamiltonian, build_hamiltonian


@dataclass
class EigenpairResult:
    """Validated ground-state eigenpair."""

    energy: float
    wavefunction: np.ndarray  # shape (ny, nx), L2-normalized with cell area
    residual_norm: float
    residual_rel: float
    grid: GridSpec
    potential: np.ndarray
    ncv: int
    tol: float
    maxiter: int
    accepted: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": float(self.energy),
            "residual_norm": float(self.residual_norm),
            "residual_rel": float(self.residual_rel),
            "grid": self.grid.to_dict(),
            "ncv": int(self.ncv),
            "tol": float(self.tol),
            "maxiter": int(self.maxiter),
            "accepted": bool(self.accepted),
            "message": self.message,
        }


def normalize_wavefunction(psi: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Normalize a finite nonzero real field in the grid-weighted L2 norm.

    Preserve the ordinary arithmetic path, with scaled normalization when the
    dimensional square would overflow/underflow. Global amplitude is arbitrary:
    a small nonzero field is not a zero field.
    """
    area = cell_area(grid)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Grid cell area must be finite and positive")
    if np.iscomplexobj(psi):
        raise ValueError("Wavefunction must be real")
    arr = np.asarray(psi, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size != grid.n_dof or not np.isfinite(flat).all():
        raise ValueError(f"Wavefunction must contain {grid.n_dof} finite values")
    scale = float(np.max(np.abs(flat)))
    if scale == 0:
        raise ValueError("Cannot normalize zero wavefunction")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norm = np.sqrt(np.sum(flat * flat) * area)
    if np.isfinite(norm) and norm >= 1e-15:
        result = flat / norm
    else:
        scaled = flat / scale
        result = (scaled / np.linalg.norm(scaled)) / np.sqrt(area)
    if not np.isfinite(result).all():
        raise ValueError("Normalized wavefunction exceeds finite float64 range")
    return result.reshape(arr.shape)


def discrete_inner(a: np.ndarray, b: np.ndarray, grid: GridSpec) -> float:
    """Discrete L2 inner product with cell area weight."""
    return float(np.sum(np.asarray(a) * np.asarray(b)) * cell_area(grid))


def residual_stats(
    ham: Hamiltonian,
    energy: float,
    psi: np.ndarray,
) -> tuple[float, float]:
    """Return (||(H-E)ψ||₂, ||(H-E)ψ||₂ / ||ψ||₂) in vector Euclidean norm."""
    flat = np.asarray(psi, dtype=np.float64).reshape(-1)
    r = ham.matvec(flat) - energy * flat
    r_norm = float(np.linalg.norm(r))
    psi_norm = float(np.linalg.norm(flat))
    rel = r_norm / max(psi_norm, 1e-15)
    return r_norm, rel


def solve_ground_state(
    ham: Hamiltonian,
    *,
    tol: float = 1e-8,
    maxiter: int = 5000,
    ncv: int | None = None,
    residual_tol: float = 1e-5,
    reject_on_failure: bool = True,
    v0: np.ndarray | None = None,
) -> EigenpairResult:
    """Compute the lowest eigenpair of a sparse symmetric Hamiltonian.

    Uses ``scipy.sparse.linalg.eigsh`` with ``which='SA'``.
    """
    n = ham.n_dof
    k = 1
    ncv_use = ncv if ncv is not None else min(max(2 * k + 1, 20), n)

    v0_use = None
    if v0 is not None:
        v0_use = np.asarray(v0, dtype=np.float64).reshape(-1).copy()
        if v0_use.size != n:
            raise ValueError(f"v0 has {v0_use.size} entries; expected {n}")
        if not np.all(np.isfinite(v0_use)):
            raise ValueError("v0 must contain only finite values")
        v0_norm = float(np.linalg.norm(v0_use))
        if v0_norm <= 1e-15:
            raise ValueError("v0 must be nonzero")
        v0_use /= v0_norm

    try:
        evals, evecs = eigsh(
            ham.matrix,
            k=k,
            which="SA",
            tol=tol,
            maxiter=maxiter,
            ncv=ncv_use,
            v0=v0_use,
            return_eigenvectors=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface solver failures
        if reject_on_failure:
            raise RuntimeError(f"eigsh failed: {exc}") from exc
        return EigenpairResult(
            energy=float("nan"),
            wavefunction=np.full((ham.grid.ny, ham.grid.nx), np.nan),
            residual_norm=float("nan"),
            residual_rel=float("nan"),
            grid=ham.grid,
            potential=ham.potential,
            ncv=ncv_use,
            tol=tol,
            maxiter=maxiter,
            accepted=False,
            message=str(exc),
        )

    energy = float(evals[0])
    psi = normalize_wavefunction(evecs[:, 0].reshape(ham.grid.ny, ham.grid.nx), ham.grid)
    # Canonicalize global sign: make the max-|entry| positive
    flat = psi.reshape(-1)
    idx = int(np.argmax(np.abs(flat)))
    if flat[idx] < 0:
        psi = -psi

    r_norm, r_rel = residual_stats(ham, energy, psi)
    accepted = bool(np.isfinite(energy) and np.all(np.isfinite(psi)) and r_rel <= residual_tol)
    message = "" if accepted else f"Residual relative {r_rel:.3e} exceeds tol {residual_tol:.3e}"
    if reject_on_failure and not accepted:
        raise RuntimeError(f"Eigenpair rejected: {message}")

    return EigenpairResult(
        energy=energy,
        wavefunction=psi,
        residual_norm=r_norm,
        residual_rel=r_rel,
        grid=ham.grid,
        potential=ham.potential.copy(),
        ncv=ncv_use,
        tol=tol,
        maxiter=maxiter,
        accepted=accepted,
        message=message,
    )


def solve_ground_state_from_potential(
    potential: np.ndarray,
    grid: GridSpec,
    **kwargs: Any,
) -> EigenpairResult:
    """Convenience: build H from V and solve ground state."""
    ham = build_hamiltonian(grid, potential)
    return solve_ground_state(ham, **kwargs)
