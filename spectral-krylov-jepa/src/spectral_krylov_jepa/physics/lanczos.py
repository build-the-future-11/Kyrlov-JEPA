"""Numerically stable short Lanczos / Krylov trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectral_krylov_jepa.physics.hamiltonian import Hamiltonian


@dataclass
class LanczosResult:
    """Short Lanczos trajectory of depth K (K+1 vectors q0..qK)."""

    q: np.ndarray  # shape (K+1, n_dof)
    alpha: np.ndarray  # shape (K,)  α_0 .. α_{K-1} for steps producing q1..qK
    beta: np.ndarray  # shape (K,)  β_1 .. β_K
    depth: int
    q0_seed: int
    reorthogonalize: bool
    breakdown: bool
    message: str = ""

    @property
    def vectors(self) -> list[np.ndarray]:
        return [self.q[i] for i in range(self.q.shape[0])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": int(self.depth),
            "q0_seed": int(self.q0_seed),
            "reorthogonalize": bool(self.reorthogonalize),
            "breakdown": bool(self.breakdown),
            "message": self.message,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
        }


def _normalize(v: np.ndarray, eps: float = 1e-14) -> tuple[np.ndarray, float]:
    nrm = float(np.linalg.norm(v))
    if nrm < eps:
        return v, nrm
    return v / nrm, nrm


def random_normalized_start(
    n_dof: int,
    seed: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a unit Euclidean-norm start vector."""
    rng = rng or np.random.default_rng(seed)
    q0 = rng.standard_normal(n_dof).astype(np.float64)
    q0, nrm = _normalize(q0)
    if nrm < 1e-14:
        raise RuntimeError("Failed to sample non-zero start vector")
    return q0


def run_lanczos(
    ham: Hamiltonian,
    q0: np.ndarray | None = None,
    *,
    depth: int = 3,
    q0_seed: int = 0,
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-12,
    max_restarts: int = 5,
) -> LanczosResult:
    """Run a short Lanczos iteration with optional full reorthogonalization.

    Recurrence (j = 0..depth-1):
        w = H q_j - β_j q_{j-1}
        α_j = q_jᵀ w
        w = w - α_j q_j
        [optional: reorthogonalize against q_0..q_j]
        β_{j+1} = ||w||
        q_{j+1} = w / β_{j+1}

    Stores q_0..q_depth, α_0..α_{depth-1}, β_1..β_depth.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    n = ham.n_dof
    last_error = ""
    for restart in range(max_restarts):
        seed = q0_seed + restart * 10_007
        if q0 is None or restart > 0:
            start = random_normalized_start(n, seed)
        else:
            start = np.asarray(q0, dtype=np.float64).reshape(-1)
            start, nrm = _normalize(start)
            if nrm < breakdown_tol:
                last_error = "q0 near zero"
                continue

        qs = np.zeros((depth + 1, n), dtype=np.float64)
        alphas = np.zeros(depth, dtype=np.float64)
        betas = np.zeros(depth, dtype=np.float64)
        qs[0] = start
        beta_prev = 0.0
        breakdown = False

        for j in range(depth):
            w = ham.matvec(qs[j])
            if j > 0:
                w = w - beta_prev * qs[j - 1]
            alpha = float(np.dot(qs[j], w))
            w = w - alpha * qs[j]
            if reorthogonalize:
                # Full reorthogonalization against q_0..q_j
                for i in range(j + 1):
                    w = w - np.dot(qs[i], w) * qs[i]
            beta = float(np.linalg.norm(w))
            alphas[j] = alpha
            betas[j] = beta
            if beta < breakdown_tol:
                breakdown = True
                last_error = f"Lanczos breakdown at j={j}, beta={beta:.3e}"
                break
            qs[j + 1] = w / beta
            beta_prev = beta

        if breakdown:
            continue

        return LanczosResult(
            q=qs,
            alpha=alphas,
            beta=betas,
            depth=depth,
            q0_seed=seed if (q0 is None or restart > 0) else q0_seed,
            reorthogonalize=reorthogonalize,
            breakdown=False,
            message="",
        )

    raise RuntimeError(f"Lanczos failed after {max_restarts} restarts: {last_error}")


def lanczos_orthogonality_error(result: LanczosResult) -> float:
    """Max |q_iᵀ q_j - δ_ij| over stored vectors."""
    q = result.q
    g = q @ q.T
    target = np.eye(q.shape[0], dtype=np.float64)
    return float(np.max(np.abs(g - target)))


def lanczos_recurrence_residual(ham: Hamiltonian, result: LanczosResult) -> float:
    """Max relative residual of the three-term recurrence at each step."""
    q = result.q
    alpha = result.alpha
    beta = result.beta
    depth = result.depth
    max_res = 0.0
    for j in range(depth):
        # H q_j = β_j q_{j-1} + α_j q_j + β_{j+1} q_{j+1}
        hq = ham.matvec(q[j])
        recon = alpha[j] * q[j] + beta[j] * q[j + 1]
        if j > 0:
            recon = recon + beta[j - 1] * q[j - 1]
        denom = max(float(np.linalg.norm(hq)), 1e-15)
        max_res = max(max_res, float(np.linalg.norm(hq - recon)) / denom)
    return max_res


def validate_lanczos(
    ham: Hamiltonian,
    result: LanczosResult,
    *,
    ortho_tol: float = 1e-6,
    recurrence_tol: float = 1e-5,
    norm_tol: float = 1e-6,
) -> dict[str, Any]:
    """Validate norms, orthogonality, recurrence, and finiteness."""
    norms = np.linalg.norm(result.q, axis=1)
    ortho = lanczos_orthogonality_error(result)
    rec = lanczos_recurrence_residual(ham, result)
    finite = bool(np.all(np.isfinite(result.q)) and np.all(np.isfinite(result.alpha)) and np.all(np.isfinite(result.beta)))
    ok = (
        finite
        and float(np.max(np.abs(norms - 1.0))) <= norm_tol
        and ortho <= ortho_tol
        and rec <= recurrence_tol
        and not result.breakdown
        and bool(np.all(result.beta > 0))
    )
    return {
        "ok": ok,
        "finite": finite,
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "orthogonality_error": ortho,
        "recurrence_residual": rec,
        "min_beta": float(np.min(result.beta)),
    }
