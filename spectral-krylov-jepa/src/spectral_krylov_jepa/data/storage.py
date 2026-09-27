"""HDF5 storage for unlabeled and labeled datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _encode_meta(meta: dict[str, Any]) -> bytes:
    return json.dumps(meta, sort_keys=True, default=str).encode("utf-8")


def _decode_meta(raw: bytes | str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def create_unlabeled_store(
    path: str | Path,
    *,
    n_examples: int,
    n_starts: int,
    depth: int,
    n_dof: int,
    ny: int,
    nx: int,
    compression: str | None = "gzip",
) -> h5py.File:
    """Create an HDF5 file for unlabeled Krylov pretraining data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["n_examples"] = n_examples
    f.attrs["n_starts"] = n_starts
    f.attrs["depth"] = depth
    f.attrs["n_dof"] = n_dof
    f.attrs["ny"] = ny
    f.attrs["nx"] = nx
    f.attrs["kind"] = "unlabeled"

    kw: dict[str, Any] = {"compression": compression} if compression else {}
    f.create_dataset("potential", shape=(n_examples, ny, nx), dtype="float32", **kw)
    f.create_dataset("q", shape=(n_examples, n_starts, depth + 1, n_dof), dtype="float32", **kw)
    f.create_dataset("alpha", shape=(n_examples, n_starts, depth), dtype="float32", **kw)
    f.create_dataset("beta", shape=(n_examples, n_starts, depth), dtype="float32", **kw)
    dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("meta", shape=(n_examples,), dtype=dt)
    return f


def write_unlabeled_example(
    f: h5py.File,
    index: int,
    *,
    potential: np.ndarray,
    q: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    meta: dict[str, Any],
) -> None:
    f["potential"][index] = np.asarray(potential, dtype=np.float32)
    f["q"][index] = np.asarray(q, dtype=np.float32)
    f["alpha"][index] = np.asarray(alpha, dtype=np.float32)
    f["beta"][index] = np.asarray(beta, dtype=np.float32)
    f["meta"][index] = _encode_meta(meta)


def create_labeled_store(
    path: str | Path,
    *,
    n_examples: int,
    ny: int,
    nx: int,
    compression: str | None = "gzip",
) -> h5py.File:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["n_examples"] = n_examples
    f.attrs["ny"] = ny
    f.attrs["nx"] = nx
    f.attrs["kind"] = "labeled"
    kw: dict[str, Any] = {"compression": compression} if compression else {}
    f.create_dataset("potential", shape=(n_examples, ny, nx), dtype="float32", **kw)
    f.create_dataset("psi0", shape=(n_examples, ny, nx), dtype="float32", **kw)
    f.create_dataset("energy", shape=(n_examples,), dtype="float64")
    f.create_dataset("residual_rel", shape=(n_examples,), dtype="float64")
    dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("meta", shape=(n_examples,), dtype=dt)
    return f


def write_labeled_example(
    f: h5py.File,
    index: int,
    *,
    potential: np.ndarray,
    psi0: np.ndarray,
    energy: float,
    residual_rel: float,
    meta: dict[str, Any],
) -> None:
    f["potential"][index] = np.asarray(potential, dtype=np.float32)
    f["psi0"][index] = np.asarray(psi0, dtype=np.float32)
    f["energy"][index] = float(energy)
    f["residual_rel"][index] = float(residual_rel)
    f["meta"][index] = _encode_meta(meta)


def read_meta(f: h5py.File, index: int) -> dict[str, Any]:
    return _decode_meta(f["meta"][index])
