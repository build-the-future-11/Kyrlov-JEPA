"""Evaluation metrics for Darcy surrogates."""

from __future__ import annotations

from typing import Any

import numpy as np

from lmop_jepa.physics import DarcyGrid, apply_operator, build_darcy_matrix


def relative_l2(u_hat: np.ndarray, u: np.ndarray) -> float:
    num = np.linalg.norm((u_hat - u).ravel())
    den = np.linalg.norm(u.ravel()) + 1e-30
    return float(num / den)


def h1_seminorm_error(u_hat: np.ndarray, u: np.ndarray, h: float) -> float:
    def grads(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vx = np.gradient(v, h, axis=1)
        vy = np.gradient(v, h, axis=0)
        return vx, vy

    gx_h, gy_h = grads(u_hat)
    gx, gy = grads(u)
    num = np.sqrt(np.sum((gx_h - gx) ** 2 + (gy_h - gy) ** 2))
    den = np.sqrt(np.sum(gx**2 + gy**2)) + 1e-30
    return float(num / den)


def energy_norm_error(u_hat: np.ndarray, u: np.ndarray, a: np.ndarray, grid: DarcyGrid) -> float:
    A = build_darcy_matrix(a, grid)
    e = (u_hat - u).ravel()
    ur = u.ravel()
    num = float(e @ (A @ e))
    den = float(ur @ (A @ ur)) + 1e-30
    return float(np.sqrt(max(num, 0.0) / den))


def flux_error(u_hat: np.ndarray, u: np.ndarray, a: np.ndarray, h: float) -> float:
    def flux(v: np.ndarray) -> np.ndarray:
        vx = np.gradient(v, h, axis=1)
        vy = np.gradient(v, h, axis=0)
        return np.stack([-a * vx, -a * vy], axis=0)

    qh = flux(u_hat)
    q = flux(u)
    num = np.linalg.norm((qh - q).ravel())
    den = np.linalg.norm(q.ravel()) + 1e-30
    return float(num / den)


def pde_residual(u_hat: np.ndarray, a: np.ndarray, f: np.ndarray, grid: DarcyGrid) -> float:
    r = apply_operator(a, u_hat, grid) - f
    return float(np.linalg.norm(r.ravel()) / (np.linalg.norm(f.ravel()) + 1e-30))


def evaluate_example(
    a: np.ndarray,
    f: np.ndarray,
    u: np.ndarray,
    u_hat: np.ndarray,
    grid: DarcyGrid,
) -> dict[str, float]:
    return {
        "relative_l2": relative_l2(u_hat, u),
        "h1_error": h1_seminorm_error(u_hat, u, grid.h),
        "energy_error": energy_norm_error(u_hat, u, a, grid),
        "flux_error": flux_error(u_hat, u, a, grid.h),
        "pde_residual": pde_residual(u_hat, a, f, grid),
        "mse": float(np.mean((u_hat - u) ** 2)),
    }


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        samples[i] = v[rng.integers(0, n, size=n)].mean()
    return {
        "estimate": float(v.mean()),
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "n": n,
        "n_resamples": n_resamples,
    }
