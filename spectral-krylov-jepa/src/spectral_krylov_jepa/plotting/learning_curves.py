"""Label-efficiency and learning-curve figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_label_efficiency(
    table: list[dict[str, Any]],
    output_path: str | Path,
    *,
    y_key: str = "fidelity_mean",
    yerr_key: str = "fidelity_std",
    title: str = "Label efficiency",
) -> Path:
    """Figure B: fidelity vs number of labeled Hamiltonians."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = sorted({r["method"] for r in table})
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for method in methods:
        rows = sorted([r for r in table if r["method"] == method], key=lambda r: r["n_labels"])
        xs = [r["n_labels"] for r in rows]
        ys = [r[y_key] for r in rows]
        yerr = [r.get(yerr_key, 0.0) for r in rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=method, capsize=3)
    ax.set_xlabel("Number of solved Hamiltonians")
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
