"""Paired bootstrap confidence intervals over test examples."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray | None = None,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Bootstrap CI for mean(values_a) or mean(values_a - values_b) if paired.

    Training-seed uncertainty is NOT mixed into this estimate — only
    finite-test-set uncertainty over examples.
    """
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    if values_b is not None:
        b = np.asarray(values_b, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("Paired arrays must have the same shape")
        data = a - b
        name = "mean_diff"
    else:
        data = a
        name = "mean"

    stat = statistic or (lambda x: float(np.mean(x)))
    rng = np.random.default_rng(seed)
    n = data.size
    if n == 0:
        return {"statistic": name, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = stat(data[idx])

    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return {
        "statistic": name,
        "estimate": float(stat(data)),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n": n,
    }
