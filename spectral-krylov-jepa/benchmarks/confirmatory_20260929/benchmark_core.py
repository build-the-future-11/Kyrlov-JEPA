from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from functools import lru_cache
import hashlib
import json
import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class Grid:
    n: int = 16
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    @property
    def h(self) -> float:
        return (self.x_max - self.x_min) / (self.n + 1)

    @property
    def n_dof(self) -> int:
        return self.n * self.n

    @property
    def cell_area(self) -> float:
        return self.h * self.h

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + self.h * np.arange(1, self.n + 1, dtype=np.float64)
        y = self.y_min + self.h * np.arange(1, self.n + 1, dtype=np.float64)
        return x, y


@dataclass(frozen=True)
class ComputeAccounting:
    method: str
    uses_labels: bool
    uses_unlabeled_pretraining: bool
    uses_query_hamiltonian: bool
    query_full_h_matvecs: int
    reusable_setup_h_matvecs: int
    basis_dimension: int
    lanczos_depth: int
    pretrain_optimizer_steps: int
    finetune_epoch_cap: int
    notes: str


@dataclass
class Prediction:
    energy: float
    psi: np.ndarray
    accounting: ComputeAccounting
    elapsed_sec: float


def _laplacian_2d(grid: Grid) -> sparse.csr_matrix:
    n = grid.n
    h2 = grid.h * grid.h
    main = np.full(n, -2.0 / h2)
    off = np.full(n - 1, 1.0 / h2)
    one = sparse.diags((off, main, off), (-1, 0, 1), format="csr")
    eye = sparse.eye(n, format="csr")
    return (sparse.kron(eye, one, format="csr") + sparse.kron(one, eye, format="csr")).tocsr()


def hamiltonian(potential: np.ndarray, grid: Grid) -> sparse.csr_matrix:
    v = np.asarray(potential, dtype=np.float64)
    if v.shape != (grid.n, grid.n):
        raise ValueError(f"potential shape {v.shape} != {(grid.n, grid.n)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("potential contains non-finite values")
    kinetic = (-0.5) * _laplacian_2d(grid)
    return (kinetic + sparse.diags(v.reshape(-1), 0, format="csr")).tocsr()


def normalize(vec: np.ndarray, grid: Grid) -> np.ndarray:
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = math.sqrt(float(np.dot(flat, flat)) * grid.cell_area)
    if norm < 1e-15:
        raise ValueError("cannot normalize near-zero state")
    return (flat / norm).reshape(grid.n, grid.n)


def sine_basis(grid: Grid, side: int) -> np.ndarray:
    if not 1 <= side <= grid.n:
        raise ValueError("side must be between 1 and grid.n")
    x, y = grid.coords()
    x = (x - grid.x_min) / (grid.x_max - grid.x_min)
    y = (y - grid.y_min) / (grid.y_max - grid.y_min)
    cols = []
    for my in range(1, side + 1):
        for mx in range(1, side + 1):
            q = np.outer(np.sin(np.pi * my * y), np.sin(np.pi * mx * x)).reshape(-1)
            q = q / np.linalg.norm(q)
            cols.append(q)
    return np.stack(cols, axis=1)


def exact_ground_state(potential: np.ndarray, grid: Grid) -> tuple[float, np.ndarray]:
    h = hamiltonian(potential, grid)
    vals, vecs = eigsh(
        h,
        k=1,
        which="SA",
        tol=1e-10,
        maxiter=5000,
        v0=np.ones(grid.n_dof, dtype=np.float64),
    )
    psi = normalize(vecs[:, 0], grid)
    flat = psi.reshape(-1)
    idx = int(np.argmax(np.abs(flat)))
    if flat[idx] < 0:
        psi = -psi
    return float(vals[0]), psi


def fixed_ritz(potential: np.ndarray, grid: Grid, side: int = 3) -> Prediction:
    start = time.perf_counter()
    b = sine_basis(grid, side)
    h = hamiltonian(potential, grid)
    hb = h @ b
    hp = (b.T @ hb)
    hp = (hp + hp.T) / 2.0
    vals, vecs = np.linalg.eigh(hp)
    psi = normalize(b @ vecs[:, 0], grid)
    return Prediction(
        energy=float(vals[0]),
        psi=psi,
        accounting=ComputeAccounting(
            method=f"fixed_ritz_{side}x{side}",
            uses_labels=False,
            uses_unlabeled_pretraining=False,
            uses_query_hamiltonian=True,
            query_full_h_matvecs=side * side,
            reusable_setup_h_matvecs=0,
            basis_dimension=side * side,
            lanczos_depth=0,
            pretrain_optimizer_steps=0,
            finetune_epoch_cap=0,
            notes="Exact Rayleigh-Ritz in fixed discrete sine basis; H access at inference.",
        ),
        elapsed_sec=time.perf_counter() - start,
    )


@lru_cache(maxsize=32)
def _pt_reference(n: int, x_min: float, x_max: float, y_min: float, y_max: float, side: int) -> tuple[np.ndarray, np.ndarray]:
    grid = Grid(n=n, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    b = sine_basis(grid, side)
    h0 = hamiltonian(np.zeros((grid.n, grid.n), dtype=np.float64), grid)
    h0b = h0 @ b
    e0_mat = b.T @ h0b
    e0 = np.diag((e0_mat + e0_mat.T) / 2.0).copy()
    b.setflags(write=False)
    e0.setflags(write=False)
    return b, e0


def pt2(potential: np.ndarray, grid: Grid, side: int = 3) -> Prediction:
    start = time.perf_counter()
    b, e0 = _pt_reference(grid.n, grid.x_min, grid.x_max, grid.y_min, grid.y_max, side)
    vflat = np.asarray(potential, dtype=np.float64).reshape(-1)
    vmat = b.T @ (vflat[:, None] * b)
    coeff = np.zeros(b.shape[1], dtype=np.float64)
    coeff[0] = 1.0
    e2 = 0.0
    for j in range(1, b.shape[1]):
        denom = float(e0[0] - e0[j])
        if abs(denom) < 1e-12:
            continue
        amp = float(vmat[j, 0] / denom)
        coeff[j] = amp
        e2 += float((vmat[0, j] * vmat[j, 0]) / denom)
    energy = float(e0[0] + vmat[0, 0] + e2)
    psi = normalize(b @ coeff, grid)
    return Prediction(
        energy=energy,
        psi=psi,
        accounting=ComputeAccounting(
            method=f"pt2_{side}x{side}", uses_labels=False, uses_unlabeled_pretraining=False,
            uses_query_hamiltonian=False, query_full_h_matvecs=0,
            reusable_setup_h_matvecs=side * side, basis_dimension=side * side,
            lanczos_depth=0, pretrain_optimizer_steps=0, finetune_epoch_cap=0,
            notes="PT2 energy plus first-order wavefunction in fixed sine basis; query work is B^T V B, while H0 basis projection is reusable.",
        ), elapsed_sec=time.perf_counter() - start,
    )


def _lanczos_basis(h: sparse.csr_matrix, q0: np.ndarray, depth: int, *, reorthogonalize: bool = True, breakdown_tol: float = 1e-12) -> np.ndarray:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    q = np.asarray(q0, dtype=np.float64).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-15)
    qs = [q]
    beta_prev = 0.0
    q_prev = np.zeros_like(q)
    for j in range(depth):
        cur = qs[-1]
        w = h @ cur
        if j > 0:
            w = w - beta_prev * q_prev
        alpha = float(np.dot(cur, w))
        w = w - alpha * cur
        if reorthogonalize:
            for qi in qs:
                w = w - float(np.dot(qi, w)) * qi
        beta = float(np.linalg.norm(w))
        if beta < breakdown_tol:
            break
        q_prev = cur
        beta_prev = beta
        qs.append(w / beta)
    use = qs[: min(depth, len(qs))]
    if len(use) == 0:
        raise RuntimeError("Lanczos basis construction failed")
    return np.stack(use, axis=1)


def perturbation_ritz_hybrid(potential: np.ndarray, grid: Grid, *, perturb_side: int = 3, krylov_depth: int = 3) -> Prediction:
    start = time.perf_counter()
    p0 = pt2(potential, grid, side=perturb_side)
    h = hamiltonian(potential, grid)
    q = _lanczos_basis(h, p0.psi.reshape(-1), krylov_depth, reorthogonalize=True)
    hp = q.T @ (h @ q)
    hp = (hp + hp.T) / 2.0
    vals, vecs = np.linalg.eigh(hp)
    psi = normalize(q @ vecs[:, 0], grid)
    return Prediction(
        energy=float(vals[0]), psi=psi,
        accounting=ComputeAccounting(
            method=f"pt2_to_krylov_ritz_d{krylov_depth}", uses_labels=False,
            uses_unlabeled_pretraining=False, uses_query_hamiltonian=True,
            query_full_h_matvecs=2 * krylov_depth,
            reusable_setup_h_matvecs=perturb_side * perturb_side,
            basis_dimension=krylov_depth, lanczos_depth=krylov_depth,
            pretrain_optimizer_steps=0, finetune_epoch_cap=0,
            notes="PT2/first-order perturbative start, full-reorthogonalized Lanczos, then Rayleigh-Ritz. query_full_h_matvecs counts d Lanczos applications plus d H@Q applications.",
        ), elapsed_sec=time.perf_counter() - start,
    )


def metrics(potential: np.ndarray, pred: Prediction, true_e: float, true_psi: np.ndarray, grid: Grid) -> dict[str, Any]:
    h = hamiltonian(potential, grid)
    p = pred.psi.reshape(-1)
    t = np.asarray(true_psi, dtype=np.float64).reshape(-1)
    area = grid.cell_area
    fidelity = float(abs(np.dot(p, t) * area) ** 2)
    rayleigh = float(np.dot(p, h @ p) / max(np.dot(p, p), 1e-30))
    residual_true = float(np.linalg.norm(h @ p - true_e * p) / max(np.linalg.norm(p), 1e-15))
    residual_rayleigh = float(np.linalg.norm(h @ p - rayleigh * p) / max(np.linalg.norm(p), 1e-15))
    return {
        "fidelity": fidelity,
        "rel_energy_error": float(abs(pred.energy - true_e) / (abs(true_e) + 1e-6)),
        "rel_rayleigh_energy_error": float(abs(rayleigh - true_e) / (abs(true_e) + 1e-6)),
        "residual_true_e": residual_true,
        "residual_rayleigh": residual_rayleigh,
        "elapsed_sec": float(pred.elapsed_sec),
        "compute": asdict(pred.accounting),
    }


def config_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()
