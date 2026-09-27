"""No-label sine-subspace Rayleigh--Ritz controls for the unit square."""
from __future__ import annotations

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


def sine_ritz(potential: np.ndarray, grid: GridSpec, modes: int = 1):
    """Project H into m x m discrete sine modes; m=1 is the free-box state.

    Uses the query potential, no labels or fitted statistics. Includes H access
    at inference, so its cost must not be presented as neural-surrogate cost.
    """
    if not 1 <= modes <= grid.n_interior:
        raise ValueError("modes must be between 1 and the grid size")
    x = (grid.x_coords() - grid.x_min) / (grid.x_max - grid.x_min)
    y = (grid.y_coords() - grid.y_min) / (grid.y_max - grid.y_min)
    basis = np.stack([
        np.outer(np.sin(np.pi * j * y), np.sin(np.pi * i * x)).ravel()
        for j in range(1, modes + 1) for i in range(1, modes + 1)
    ], axis=1)
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    ham = build_hamiltonian(grid, potential)
    h_basis = np.column_stack([ham.matvec(q) for q in basis.T])
    projected = basis.T @ h_basis
    energies, vectors = np.linalg.eigh((projected + projected.T) / 2)
    psi = basis @ vectors[:, 0] / np.sqrt(cell_area(grid))
    return float(energies[0]), psi.reshape(grid.ny, grid.nx)
