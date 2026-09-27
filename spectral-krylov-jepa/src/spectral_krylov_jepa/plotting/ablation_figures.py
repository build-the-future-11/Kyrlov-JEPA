"""Ablation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_krylov_depth_ablation(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Figure D: Krylov depth vs downstream fidelity."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    xs = [r["context_steps"] for r in rows]
    ys = [r["fidelity_mean"] for r in rows]
    yerr = [r.get("fidelity_std", 0.0) for r in rows]
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
    ax.set_xlabel("Krylov context depth K")
    ax.set_ylabel("Downstream fidelity")
    ax.set_title("Krylov-depth ablation")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
