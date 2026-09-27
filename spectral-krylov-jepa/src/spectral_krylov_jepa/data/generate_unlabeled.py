"""Generate unlabeled pretraining corpora (potentials + Lanczos trajectories)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from spectral_krylov_jepa.data.storage import create_unlabeled_store, write_unlabeled_example
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.lanczos import run_lanczos, validate_lanczos
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.utils.io import write_json
from spectral_krylov_jepa.utils.logging import setup_logger


def generate_unlabeled_dataset(
    *,
    output_path: str | Path,
    n_examples: int = 1000,
    n_starts: int = 2,
    depth: int = 3,
    grid: GridSpec | None = None,
    family: str = "id_gaussian_mixture",
    base_seed: int = 10_000,
    shuffle_physics: bool = False,
    shuffle_seed: int = 0,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate unlabeled Krylov trajectories without eigensolves.

    If ``shuffle_physics`` is True, pair each potential V_i with a trajectory
    computed under V_{π(i)} (critical negative control).
    """
    logger = setup_logger("generate_unlabeled")
    grid = grid or GridSpec()
    output_path = Path(output_path)

    potentials: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    trajs: list[dict[str, Any]] = []

    logger.info(
        "Generating %d unlabeled potentials (family=%s, grid=%d, depth=%d, starts=%d)",
        n_examples,
        family,
        grid.n_interior,
        depth,
        n_starts,
    )

    for i in tqdm(range(n_examples), desc="unlabeled potentials"):
        seed = base_seed + i
        v, spec = generate_potential(family, seed, grid=grid)
        potentials.append(v)
        metas.append(spec.to_dict())

    # Optionally shuffle which Hamiltonian generates trajectories
    traj_indices = np.arange(n_examples)
    if shuffle_physics:
        rng = np.random.default_rng(shuffle_seed)
        traj_indices = rng.permutation(traj_indices)

    for i in tqdm(range(n_examples), desc="lanczos"):
        ham_idx = int(traj_indices[i])
        ham = build_hamiltonian(grid, potentials[ham_idx])
        q_all = np.zeros((n_starts, depth + 1, grid.n_dof), dtype=np.float64)
        a_all = np.zeros((n_starts, depth), dtype=np.float64)
        b_all = np.zeros((n_starts, depth), dtype=np.float64)
        start_meta = []
        for s in range(n_starts):
            lanc = run_lanczos(ham, depth=depth, q0_seed=base_seed + i * 97 + s * 13)
            val = validate_lanczos(ham, lanc)
            if not val["ok"]:
                raise RuntimeError(f"Lanczos validation failed at example {i}, start {s}: {val}")
            q_all[s] = lanc.q
            a_all[s] = lanc.alpha
            b_all[s] = lanc.beta
            start_meta.append(lanc.to_dict())
        trajs.append(
            {
                "q": q_all,
                "alpha": a_all,
                "beta": b_all,
                "ham_index": ham_idx,
                "starts": start_meta,
            }
        )

    f = create_unlabeled_store(
        output_path,
        n_examples=n_examples,
        n_starts=n_starts,
        depth=depth,
        n_dof=grid.n_dof,
        ny=grid.ny,
        nx=grid.nx,
    )
    try:
        for i in range(n_examples):
            meta = {
                "potential": metas[i],
                "ham_index": int(trajs[i]["ham_index"]),
                "shuffle_physics": bool(shuffle_physics),
                "starts": trajs[i]["starts"],
                "index": i,
            }
            write_unlabeled_example(
                f,
                i,
                potential=potentials[i],
                q=trajs[i]["q"],
                alpha=trajs[i]["alpha"],
                beta=trajs[i]["beta"],
                meta=meta,
            )
    finally:
        f.close()

    manifest = {
        "path": str(output_path),
        "n_examples": n_examples,
        "n_starts": n_starts,
        "depth": depth,
        "family": family,
        "base_seed": base_seed,
        "shuffle_physics": shuffle_physics,
        "shuffle_seed": shuffle_seed,
        "grid": grid.to_dict(),
        "seeds": [base_seed + i for i in range(n_examples)],
    }
    if manifest_path is not None:
        write_json(manifest, manifest_path)
    logger.info("Wrote unlabeled dataset to %s", output_path)
    return manifest
