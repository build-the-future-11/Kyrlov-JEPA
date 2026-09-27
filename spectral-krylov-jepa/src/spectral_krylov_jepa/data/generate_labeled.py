"""Generate labeled eigenstate datasets with frozen splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from spectral_krylov_jepa.data.manifests import save_dataset_manifest
from spectral_krylov_jepa.data.splits import freeze_labeled_manifest
from spectral_krylov_jepa.data.storage import create_labeled_store, write_labeled_example
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state_from_potential
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.utils.io import write_json
from spectral_krylov_jepa.utils.logging import setup_logger


def _generate_block(
    *,
    family: str,
    n: int,
    base_seed: int,
    grid: GridSpec,
    residual_tol: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], list[dict[str, Any]]]:
    potentials: list[np.ndarray] = []
    psis: list[np.ndarray] = []
    energies: list[float] = []
    residuals: list[float] = []
    metas: list[dict[str, Any]] = []
    for i in tqdm(range(n), desc=f"labeled:{family}"):
        seed = base_seed + i
        v, spec = generate_potential(family, seed, grid=grid)
        gs = solve_ground_state_from_potential(v, grid, residual_tol=residual_tol)
        potentials.append(v)
        psis.append(gs.wavefunction)
        energies.append(gs.energy)
        residuals.append(gs.residual_rel)
        meta = spec.to_dict()
        meta.update(
            {
                "energy": gs.energy,
                "residual_rel": gs.residual_rel,
                "accepted": gs.accepted,
            }
        )
        metas.append(meta)
    return potentials, psis, energies, residuals, metas


def generate_labeled_dataset(
    *,
    output_path: str | Path,
    grid: GridSpec | None = None,
    n_train: int = 120,
    n_val: int = 40,
    n_test_id: int = 40,
    n_ood_each: int = 40,
    subset_sizes: list[int] | None = None,
    split_seed: int = 2026,
    id_base_seed: int = 20_000,
    ood_base_seed: int = 30_000,
    residual_tol: float = 1e-5,
    manifest_name: str = "labeled_splits",
) -> dict[str, Any]:
    """Generate labeled ID + OOD datasets and freeze split manifests."""
    logger = setup_logger("generate_labeled")
    grid = grid or GridSpec()
    subset_sizes = subset_sizes or [10, 25, 50, 100]
    output_path = Path(output_path)

    id_count = n_train + n_val + n_test_id
    n_total = id_count + 3 * n_ood_each

    all_v: list[np.ndarray] = []
    all_psi: list[np.ndarray] = []
    all_e: list[float] = []
    all_r: list[float] = []
    all_meta: list[dict[str, Any]] = []
    families: list[str] = []

    # Ensure seed ranges do not collide across families
    blocks = [
        ("id_gaussian_mixture", id_count, id_base_seed),
        ("ood_narrow", n_ood_each, ood_base_seed),
        ("ood_strong", n_ood_each, ood_base_seed + 10_000),
        ("ood_double", n_ood_each, ood_base_seed + 20_000),
    ]
    for family, n, base in blocks:
        v, psi, e, r, meta = _generate_block(
            family=family,
            n=n,
            base_seed=base,
            grid=grid,
            residual_tol=residual_tol,
        )
        all_v.extend(v)
        all_psi.extend(psi)
        all_e.extend(e)
        all_r.extend(r)
        all_meta.extend(meta)
        families.extend([family] * n)

    assert len(all_v) == n_total

    # Reject near-identical seeds across the whole corpus
    seeds = [m["seed"] for m in all_meta]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate potential seeds detected in labeled corpus")

    f = create_labeled_store(output_path, n_examples=n_total, ny=grid.ny, nx=grid.nx)
    try:
        for i in range(n_total):
            write_labeled_example(
                f,
                i,
                potential=all_v[i],
                psi0=all_psi[i],
                energy=all_e[i],
                residual_rel=all_r[i],
                meta=all_meta[i],
            )
    finally:
        f.close()

    manifest = freeze_labeled_manifest(
        n_total=n_total,
        seed=split_seed,
        n_train=n_train,
        n_val=n_val,
        n_test_id=n_test_id,
        n_ood_each=n_ood_each,
        subset_sizes=subset_sizes,
        family_by_index=families,
    )
    manifest["data_path"] = str(output_path)
    manifest["grid"] = grid.to_dict()
    manifest["id_base_seed"] = id_base_seed
    manifest["ood_base_seed"] = ood_base_seed
    manifest_path = save_dataset_manifest(manifest_name, manifest)
    write_json(manifest, Path(output_path).with_suffix(".manifest.json"))
    logger.info("Wrote labeled dataset to %s; manifest %s", output_path, manifest_path)
    return manifest
