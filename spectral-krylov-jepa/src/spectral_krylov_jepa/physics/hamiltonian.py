"""Hamiltonian construction H = -½Δ + V for 2D Schrödinger problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.laplacian import build_laplacian, laplacian_symmetry_error


@dataclass
class Hamiltonian:
    """Sparse Hamiltonian on the interior grid."""

    matrix: sparse.csr_matrix
    grid: GridSpec
    potential: np.ndarray  # shape (ny, nx)

    @property
    def n_dof(self) -> int:
        return self.grid.n_dof

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(x, dtype=np.float64).reshape(-1)

    def to_dense(self) -> np.ndarray:
        return self.matrix.toarray()


def build_hamiltonian(grid: GridSpec, potential: np.ndarray) -> Hamiltonian:
    """Construct H = -½ Δ + diag(V) on the interior grid.

    Parameters
    ----------
    grid:
        Interior grid specification.
    potential:
        Potential values on interior nodes, shape (ny, nx).
    """
    v = np.asarray(potential, dtype=np.float64)
    if v.shape != (grid.ny, grid.nx):
        raise ValueError(f"Potential shape {v.shape} != {(grid.ny, grid.nx)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("Potential contains non-finite values")

    lap = build_laplacian(grid)
    kinetic = (-0.5) * lap
    potential_diag = sparse.diags(v.reshape(-1), offsets=0, shape=kinetic.shape, format="csr")
    h = (kinetic + potential_diag).tocsr()
    return Hamiltonian(matrix=h, grid=grid, potential=v.copy())


def hamiltonian_symmetry_error(ham: Hamiltonian, tol: float = 1e-12) -> float:
    """Relative infinity-norm symmetry error of H."""
    return laplacian_symmetry_error(ham.matrix, tol=tol)


def apply_hamiltonian(ham: Hamiltonian, state: np.ndarray) -> np.ndarray:
    """Apply H to a flattened or 2D state; return same shape as input."""
    flat = np.asarray(state, dtype=np.float64)
    out_shape = flat.shape
    vec = flat.reshape(-1)
    if vec.size != ham.n_dof:
        raise ValueError(f"State size {vec.size} != n_dof {ham.n_dof}")
    result = ham.matvec(vec)
    return result.reshape(out_shape)
