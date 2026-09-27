"""Physics diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian


def plot_physics_panel(
    potential: np.ndarray,
    psi0: np.ndarray,
    energy: float,
    grid: GridSpec,
    output_path: str | Path,
    residual_map: np.ndarray | None = None,
) -> Path:
    """Figure A: Potential | Ground state | |ψ|² | residual map."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if residual_map is None:
        ham = build_hamiltonian(grid, potential)
        flat = psi0.reshape(-1)
        r = (ham.matvec(flat) - energy * flat).reshape(psi0.shape)
        residual_map = r

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = [
        (potential, "Potential V", "viridis"),
        (psi0, r"Ground state $\psi_0$", "RdBu_r"),
        (np.abs(psi0) ** 2, r"$|\psi_0|^2$", "magma"),
        (residual_map, r"Residual $(H-E)\psi$", "coolwarm"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"E0 = {energy:.6f}", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_prediction_panel(
    psi_true: np.ndarray,
    psi_pred: np.ndarray,
    output_path: str | Path,
    fidelity: float | None = None,
) -> Path:
    """Figure E: true ψ | predicted ψ | absolute error."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Align sign
    if np.dot(psi_true.ravel(), psi_pred.ravel()) < 0:
        psi_pred = -psi_pred
    err = np.abs(psi_true - psi_pred)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        [psi_true, psi_pred, err],
        [r"True $\psi_0$", r"Predicted $\hat\psi_0$", r"$|\psi-\hat\psi|$"],
    ):
        im = ax.imshow(data, origin="lower", cmap="RdBu_r" if title != r"$|\psi-\hat\psi|$" else "magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if fidelity is not None:
        fig.suptitle(f"Fidelity = {fidelity:.4f}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
