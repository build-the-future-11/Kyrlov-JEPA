"""PyTorch datasets for unlabeled pretraining and labeled fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from spectral_krylov_jepa.data.storage import read_meta


class UnlabeledKrylovDataset(Dataset):
    """Lazy HDF5 dataset of (V, q0..qK, α, β) examples.

    Each __getitem__ returns one (example, start) pair flattened.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        remove_v: bool = False,
        start_index: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.remove_v = remove_v
        self.start_index = start_index
        with h5py.File(self.path, "r") as f:
            self.n_examples = int(f.attrs["n_examples"])
            self.n_starts = int(f.attrs["n_starts"])
            self.depth = int(f.attrs["depth"])
            self.ny = int(f.attrs["ny"])
            self.nx = int(f.attrs["nx"])
            self.n_dof = int(f.attrs["n_dof"])
        if start_index is not None:
            self._len = self.n_examples
        else:
            self._len = self.n_examples * self.n_starts
        self._file: h5py.File | None = None

    def _f(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.start_index is not None:
            ex = idx
            st = self.start_index
        else:
            ex = idx // self.n_starts
            st = idx % self.n_starts
        f = self._f()
        v = np.asarray(f["potential"][ex], dtype=np.float32)
        q = np.asarray(f["q"][ex, st], dtype=np.float32)  # (depth+1, n_dof)
        alpha = np.asarray(f["alpha"][ex, st], dtype=np.float32)
        beta = np.asarray(f["beta"][ex, st], dtype=np.float32)
        if self.remove_v:
            v = np.zeros_like(v)
        return {
            "potential": torch.from_numpy(v),
            "q": torch.from_numpy(q),
            "alpha": torch.from_numpy(alpha),
            "beta": torch.from_numpy(beta),
            "index": ex,
            "start": st,
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class LabeledEigenstateDataset(Dataset):
    """Labeled (V, E0, ψ0) dataset with optional index subset."""

    def __init__(
        self,
        path: str | Path,
        indices: list[int] | None = None,
    ) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.n_examples = int(f.attrs["n_examples"])
            self.ny = int(f.attrs["ny"])
            self.nx = int(f.attrs["nx"])
        self.indices = list(range(self.n_examples)) if indices is None else list(indices)
        self._file: h5py.File | None = None

    def _f(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.indices[idx]
        f = self._f()
        v = np.asarray(f["potential"][ex], dtype=np.float32)
        psi = np.asarray(f["psi0"][ex], dtype=np.float32)
        energy = float(f["energy"][ex])
        residual_rel = float(f["residual_rel"][ex])
        meta = read_meta(f, ex)
        return {
            "potential": torch.from_numpy(v),
            "psi0": torch.from_numpy(psi),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "residual_rel": residual_rel,
            "index": ex,
            "family": meta.get("family", "unknown"),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


SplitName = Literal[
    "train_pool",
    "validation",
    "test_ID",
    "test_OOD_narrow",
    "test_OOD_strong",
    "test_OOD_double",
]


def labeled_from_manifest(
    data_path: str | Path,
    manifest: dict[str, Any],
    split: str,
) -> LabeledEigenstateDataset:
    if split in manifest.get("subsets", {}):
        indices = manifest["subsets"][split]
    elif split in manifest.get("splits", {}):
        indices = manifest["splits"][split]
    else:
        raise KeyError(f"Unknown split {split!r}")
    return LabeledEigenstateDataset(data_path, indices=indices)
