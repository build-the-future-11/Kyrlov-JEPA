"""Frozen discrete Darcy reference: -div(a grad u) = f with Dirichlet u=0.

Identical operator used for manufactured forcing and genuine solves.
Cell-centered FD on the unit square with harmonic-averaged face permeabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


@dataclass(frozen=True)
class DarcyGrid:
    n: int = 64  # interior DOFs per axis
    x0: float = 0.0
    x1: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}")

    @property
    def h(self) -> float:
        return (self.x1 - self.x0) / (self.n + 1)

    @property
    def n_dof(self) -> int:
        return self.n * self.n

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + self.h * np.arange(1, self.n + 1, dtype=np.float64)
        X, Y = np.meshgrid(xs, xs, indexing="xy")
        return X, Y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_permeability(a: np.ndarray, a_min: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"permeability must be square 2D, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("permeability contains non-finite values")
    if np.any(a <= a_min):
        raise ValueError(f"permeability must be > {a_min}, min={a.min()}")
    return a


def build_darcy_matrix(a: np.ndarray, grid: DarcyGrid | None = None) -> sparse.csr_matrix:
    """Assemble SPD sparse matrix A for -div(a grad u) on interior nodes.

    Ordering: row-major, index = iy * n + ix.
    Face permeabilities use harmonic means of adjacent cell values.
    Dirichlet zeros are built into the stencil (no boundary DOFs).
    """
    a = validate_permeability(a)
    grid = grid or DarcyGrid(n=a.shape[0])
    if a.shape != (grid.n, grid.n):
        raise ValueError(f"a shape {a.shape} != ({grid.n},{grid.n})")

    n = grid.n
    h = grid.h
    inv_h2 = 1.0 / (h * h)
    nd = n * n

    # Face permeabilities (harmonic average); ghost a outside = neighbor (Dirichlet)
    a_e = np.zeros((n, n), dtype=np.float64)
    a_w = np.zeros((n, n), dtype=np.float64)
    a_n = np.zeros((n, n), dtype=np.float64)
    a_s = np.zeros((n, n), dtype=np.float64)
    for iy in range(n):
        for ix in range(n):
            ae = a[iy, ix] if ix == n - 1 else 2.0 * a[iy, ix] * a[iy, ix + 1] / (a[iy, ix] + a[iy, ix + 1])
            aw = a[iy, ix] if ix == 0 else 2.0 * a[iy, ix] * a[iy, ix - 1] / (a[iy, ix] + a[iy, ix - 1])
            an = a[iy, ix] if iy == n - 1 else 2.0 * a[iy, ix] * a[iy + 1, ix] / (a[iy, ix] + a[iy + 1, ix])
            as_ = a[iy, ix] if iy == 0 else 2.0 * a[iy, ix] * a[iy - 1, ix] / (a[iy, ix] + a[iy - 1, ix])
            a_e[iy, ix] = ae
            a_w[iy, ix] = aw
            a_n[iy, ix] = an
            a_s[iy, ix] = as_

    # Vectorized assembly via diags
    main = (a_e + a_w + a_n + a_s).ravel() * inv_h2
    # East neighbor (+1), broken at row ends
    east = np.zeros(nd - 1, dtype=np.float64)
    for iy in range(n):
        for ix in range(n - 1):
            j = iy * n + ix
            east[j] = -a_e[iy, ix] * inv_h2
    # North neighbor (+n)
    north = (-a_n[:-1, :].ravel()) * inv_h2

    A = sparse.diags(
        [main, east, east, north, north],
        [0, 1, -1, n, -n],
        shape=(nd, nd),
        format="csr",
        dtype=np.float64,
    )
    # Ensure exact symmetry numerically
    A = 0.5 * (A + A.T)
    return A.tocsr()


def apply_operator(a: np.ndarray, u: np.ndarray, grid: DarcyGrid | None = None) -> np.ndarray:
    grid = grid or DarcyGrid(n=a.shape[0])
    A = build_darcy_matrix(a, grid)
    return (A @ np.asarray(u, dtype=np.float64).ravel()).reshape(grid.n, grid.n)


def solve_darcy(
    a: np.ndarray,
    f: np.ndarray,
    grid: DarcyGrid | None = None,
    *,
    method: str = "spsolve",
    tol: float = 1e-12,
    residual_tol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve A u = f. Returns (u, info) with residual diagnostics."""
    a = validate_permeability(a)
    grid = grid or DarcyGrid(n=a.shape[0])
    f = np.asarray(f, dtype=np.float64)
    if f.shape != (grid.n, grid.n):
        raise ValueError(f"f shape {f.shape} != ({grid.n},{grid.n})")
    A = build_darcy_matrix(a, grid)
    rhs = f.ravel()
    if method == "spsolve":
        u_flat = spsolve(A, rhs)
        info_code = 0
    elif method == "cg":
        u_flat, info_code = cg(A, rhs, rtol=tol, atol=0.0, maxiter=max(1000, 20 * grid.n_dof))
        if info_code != 0:
            raise RuntimeError(f"CG failed with info={info_code}")
    else:
        raise ValueError(method)
    u = np.asarray(u_flat, dtype=np.float64).reshape(grid.n, grid.n)
    r = A @ u.ravel() - rhs
    rel = float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1e-30))
    if not np.all(np.isfinite(u)) or rel > residual_tol:
        raise RuntimeError(f"Darcy solve rejected: residual_rel={rel:.3e}")
    return u, {"residual_rel": rel, "method": method, "info": int(info_code)}


def manufactured_sine_mode(
    grid: DarcyGrid,
    kx: int,
    ky: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Dirichlet-compatible sine mode on the interior grid."""
    if kx < 1 or ky < 1:
        raise ValueError("mode indices must be >= 1")
    X, Y = grid.coords()
    return amplitude * np.sin(np.pi * kx * X) * np.sin(np.pi * ky * Y)


def sample_log_permeability(
    grid: DarcyGrid,
    seed: int,
    *,
    length_scale: float = 0.2,
    variance: float = 1.0,
    a_min: float = 0.1,
    a_max: float = 10.0,
) -> np.ndarray:
    """Approximate Gaussian random field via spectral synthesis on the grid."""
    rng = np.random.default_rng(seed)
    n = grid.n
    kx = np.fft.fftfreq(n, d=grid.h)
    ky = np.fft.fftfreq(n, d=grid.h)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX**2 + KY**2
    # Squared-exponential-like spectrum
    power = np.exp(-0.5 * (2.0 * np.pi * length_scale) ** 2 * k2)
    power[0, 0] = 0.0
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    field = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(power)).real
    field = field - field.mean()
    std = field.std()
    if std < 1e-12:
        field = np.zeros_like(field)
    else:
        field = field / std * np.sqrt(variance)
    a = np.exp(field)
    # Soft clip to [a_min, a_max] while staying positive
    a = np.clip(a, a_min, a_max)
    return a.astype(np.float64)


def matrix_symmetry_error(A: sparse.spmatrix) -> float:
    D = (A - A.T).tocsr()
    if D.nnz == 0:
        return 0.0
    return float(np.max(np.abs(D.data)) / max(1.0, float(np.max(np.abs(A.data)))))


def is_spd_probe(A: sparse.spmatrix, n_probe: int = 5, seed: int = 0) -> bool:
    """Cheap SPD probe: Cholesky on dense for small n else random Rayleigh."""
    n = A.shape[0]
    if n <= 256:
        try:
            np.linalg.cholesky(A.toarray())
            return True
        except np.linalg.LinAlgError:
            return False
    rng = np.random.default_rng(seed)
    for _ in range(n_probe):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        if float(v @ (A @ v)) <= 0:
            return False
    return True


def condition_estimate(A: sparse.spmatrix, k: int = 6) -> float:
    """Estimate cond via extreme eigenvalues (eigsh)."""
    from scipy.sparse.linalg import eigsh

    n = A.shape[0]
    k = min(k, max(1, n - 2))
    try:
        evals_max = eigsh(A, k=1, which="LM", return_eigenvectors=False)
        evals_min = eigsh(A, k=1, which="SM", return_eigenvectors=False)
        return float(abs(evals_max[0] / evals_min[0]))
    except Exception:  # noqa: BLE001
        return float("nan")


def manufactured_round_trip(
    a: np.ndarray,
    u_tilde: np.ndarray,
    grid: DarcyGrid | None = None,
    rtol: float = 1e-8,
) -> dict[str, float]:
    """Verify A^{-1}(A u) ≈ u under the same discrete operator."""
    grid = grid or DarcyGrid(n=a.shape[0])
    f = apply_operator(a, u_tilde, grid)
    u_rec, info = solve_darcy(a, f, grid, residual_tol=1e-6)
    err = float(np.linalg.norm((u_rec - u_tilde).ravel()) / max(np.linalg.norm(u_tilde.ravel()), 1e-30))
    return {
        "rel_roundtrip_error": err,
        "solver_residual_rel": info["residual_rel"],
        "ok": float(err <= rtol),
    }
