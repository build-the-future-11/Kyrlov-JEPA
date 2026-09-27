"""2D finite-difference Laplacian with Dirichlet boundary conditions."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from spectral_krylov_jepa.physics.grid import GridSpec


def build_laplacian(grid: GridSpec) -> sparse.csr_matrix:
    """Build the sparse 5-point Laplacian on interior nodes.

    For a scalar field u on the interior grid, the discrete Laplacian
    approximates ∂²u/∂x² + ∂²u/∂y² with Dirichlet zeros on the boundary.

    Returns a CSR matrix of shape (n_dof, n_dof), real and symmetric.
    Ordering is row-major: index = iy * nx + ix.
    """
    nx, ny = grid.nx, grid.ny
    hx2 = grid.hx * grid.hx
    hy2 = grid.hy * grid.hy
    n = nx * ny

    main = np.full(n, -2.0 / hx2 - 2.0 / hy2, dtype=np.float64)
    # Neighbors in x (same row)
    x_off = np.full(n - 1, 1.0 / hx2, dtype=np.float64)
    # Break connections across row boundaries
    for iy in range(ny):
        j = iy * nx + (nx - 1)
        if j < n - 1:
            x_off[j] = 0.0

    # Neighbors in y (same column, adjacent row)
    y_off = np.full(n - nx, 1.0 / hy2, dtype=np.float64)

    diags = [main, x_off, x_off, y_off, y_off]
    offsets = [0, 1, -1, nx, -nx]
    lap = sparse.diags(diags, offsets, shape=(n, n), format="csr", dtype=np.float64)
    return lap


def laplacian_symmetry_error(lap: sparse.spmatrix, tol: float = 1e-12) -> float:
    """Return ||A - Aᵀ||_∞ / max(1, ||A||_∞)."""
    a = lap.tocsr()
    diff = (a - a.T).tocsr()
    denom = max(1.0, float(np.abs(a.data).max()) if a.nnz else 1.0)
    if diff.nnz == 0:
        return 0.0
    return float(np.abs(diff.data).max()) / denom
