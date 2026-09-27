"""OOD comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_ood_comparison(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    metric: str = "fidelity_mean",
) -> Path:
    """Figure C: grouped ID vs OOD bars per method."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(next(iter(results.values())).keys()) if results else []
    methods = list(results.keys())
    x = np.arange(len(splits))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = []
        for sp in splits:
            summary = results[method][sp].get("summary", results[method][sp])
            vals.append(summary.get(metric, float("nan")))
        ax.bar(x + i * width, vals, width=width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("ID vs OOD comparison")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
