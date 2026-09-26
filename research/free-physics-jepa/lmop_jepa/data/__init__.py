"""Data generation, manifests, HDF5 stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lmop_jepa.physics import (
    DarcyGrid,
    apply_operator,
    manufactured_round_trip,
    manufactured_sine_mode,
    sample_log_permeability,
    solve_darcy,
)


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def nested_subsets(indices: list[int], sizes: list[int], seed: int) -> dict[str, list[int]]:
    sizes = sorted(set(int(s) for s in sizes))
    if max(sizes) > len(indices):
        raise ValueError("subset larger than pool")
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(indices, dtype=np.int64))
    return {f"n{s}": order[:s].tolist() for s in sizes}


def assert_disjoint(*groups: list[int]) -> None:
    seen: dict[int, int] = {}
    for gi, g in enumerate(groups):
        for x in g:
            if x in seen:
                raise AssertionError(f"ID {x} in groups {seen[x]} and {gi}")
            seen[x] = gi


def generate_genuine_dataset(
    path: Path,
    *,
    grid: DarcyGrid,
    n: int,
    base_seed: int,
    length_scale: float,
    variance: float,
    a_min: float,
    a_max: float,
    family: str = "id",
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f_field = np.ones((grid.n, grid.n), dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("a", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("f", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("u", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("residual_rel", shape=(n,), dtype="float64")
        h5.create_dataset("seed", shape=(n,), dtype="int64")
        h5.create_dataset("field_id", shape=(n,), dtype=h5py.string_dtype())
        h5.attrs["family"] = family
        h5.attrs["n"] = n
        h5.attrs["grid_n"] = grid.n
        h5.attrs["length_scale"] = length_scale
        ids = []
        for i in range(n):
            seed = base_seed + i
            a = sample_log_permeability(
                grid, seed, length_scale=length_scale, variance=variance, a_min=a_min, a_max=a_max
            )
            u, info = solve_darcy(a, f_field, grid)
            fid = f"{family}_{seed:08d}"
            h5["a"][i] = a.astype(np.float32)
            h5["f"][i] = f_field.astype(np.float32)
            h5["u"][i] = u.astype(np.float32)
            h5["residual_rel"][i] = info["residual_rel"]
            h5["seed"][i] = seed
            h5["field_id"][i] = fid
            ids.append(fid)
    meta = {
        "path": str(path),
        "n": n,
        "family": family,
        "grid": grid.to_dict(),
        "base_seed": base_seed,
        "length_scale": length_scale,
        "variance": variance,
        "field_ids": ids,
        "file_sha256": sha256_file(path),
    }
    write_json(meta, path.with_suffix(".meta.json"))
    return meta


def generate_manufactured_dataset(
    path: Path,
    *,
    grid: DarcyGrid,
    n: int,
    base_seed: int,
    family: str,
    kmax: int,
    n_modes_range: tuple[int, int],
    amplitude_range: tuple[float, float],
    length_scale: float,
    variance: float,
    a_min: float,
    a_max: float,
    roundtrip_tol: float = 1e-7,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("a", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("f", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("u", shape=(n, grid.n, grid.n), dtype="float32")
        h5.create_dataset("roundtrip_err", shape=(n,), dtype="float64")
        h5.create_dataset("seed", shape=(n,), dtype="int64")
        h5.create_dataset("field_id", shape=(n,), dtype=h5py.string_dtype())
        h5.attrs["family"] = family
        h5.attrs["kmax"] = kmax
        ids = []
        for i in range(n):
            seed = base_seed + i
            rng = np.random.default_rng(seed)
            a = sample_log_permeability(
                grid, seed + 10_000_003, length_scale=length_scale, variance=variance, a_min=a_min, a_max=a_max
            )
            n_modes = int(rng.integers(n_modes_range[0], n_modes_range[1] + 1))
            u = np.zeros((grid.n, grid.n), dtype=np.float64)
            for _ in range(n_modes):
                kx = int(rng.integers(1, kmax + 1))
                ky = int(rng.integers(1, kmax + 1))
                amp = float(rng.uniform(*amplitude_range))
                # random sign
                if rng.random() < 0.5:
                    amp = -amp
                u += manufactured_sine_mode(grid, kx, ky, amp)
            f = apply_operator(a, u, grid)
            rt = manufactured_round_trip(a, u, grid, rtol=roundtrip_tol)
            if not rt["ok"]:
                raise RuntimeError(f"Manufactured round-trip failed at i={i}: {rt}")
            fid = f"mfg_{family}_{seed:08d}"
            h5["a"][i] = a.astype(np.float32)
            h5["f"][i] = f.astype(np.float32)
            h5["u"][i] = u.astype(np.float32)
            h5["roundtrip_err"][i] = rt["rel_roundtrip_error"]
            h5["seed"][i] = seed
            h5["field_id"][i] = fid
            ids.append(fid)
    meta = {
        "path": str(path),
        "n": n,
        "family": family,
        "kmax": kmax,
        "grid": grid.to_dict(),
        "base_seed": base_seed,
        "field_ids": ids,
        "file_sha256": sha256_file(path),
    }
    write_json(meta, path.with_suffix(".meta.json"))
    return meta


def build_split_manifest(
    *,
    genuine_train_ids: list[str],
    genuine_val_ids: list[str],
    genuine_test_ids: list[str],
    genuine_ood_ids: list[str],
    manufactured_ids: list[str],
    subset_sizes: list[int],
    subset_seed: int,
) -> dict[str, Any]:
    # Index-based nested subsets over train pool indices
    train_idx = list(range(len(genuine_train_ids)))
    subsets = nested_subsets(train_idx, subset_sizes, subset_seed)
    # Disjointness of field IDs
    assert_disjoint(
        list(range(len(genuine_train_ids))),
        list(range(len(genuine_train_ids), len(genuine_train_ids) + len(genuine_val_ids))),
    )
    all_g = genuine_train_ids + genuine_val_ids + genuine_test_ids + genuine_ood_ids
    if len(all_g) != len(set(all_g)):
        raise AssertionError("Duplicate genuine field IDs across splits")
    if set(manufactured_ids) & set(all_g):
        raise AssertionError("Manufactured IDs collide with genuine IDs")
    man = {
        "genuine_train_ids": genuine_train_ids,
        "genuine_val_ids": genuine_val_ids,
        "genuine_test_id_ids": genuine_test_ids,
        "genuine_ood_ids": genuine_ood_ids,
        "manufactured_ids": manufactured_ids,
        "subsets": subsets,
        "subset_sizes": subset_sizes,
        "subset_seed": subset_seed,
    }
    blob = json.dumps(man, sort_keys=True).encode()
    man["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
    return man
