"""Split manifests and nested fine-tune subset construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_NAMES = (
    "train_pool",
    "validation",
    "test_ID",
    "test_OOD_narrow",
    "test_OOD_strong",
    "test_OOD_double",
)


def build_nested_subsets(
    train_indices: list[int],
    sizes: list[int],
    seed: int = 0,
) -> dict[str, list[int]]:
    """Build nested fine-tune subsets: N_small ⊂ N_larger.

    Indices are drawn from ``train_indices`` without replacement, with a
    fixed RNG so that larger subsets extend smaller ones.
    """
    sizes = sorted(set(int(s) for s in sizes))
    if max(sizes) > len(train_indices):
        raise ValueError(
            f"Requested subset size {max(sizes)} > train_pool size {len(train_indices)}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(train_indices, dtype=np.int64))
    out: dict[str, list[int]] = {}
    for n in sizes:
        out[f"n{n}"] = order[:n].tolist()
    return out


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_no_split_overlap(splits: dict[str, list[int]]) -> None:
    """Raise if any index appears in more than one top-level split."""
    seen: dict[int, str] = {}
    for name, idxs in splits.items():
        if name.startswith("n"):
            continue  # nested subsets intentionally overlap train_pool
        for i in idxs:
            if i in seen and seen[i] != name:
                raise AssertionError(f"Index {i} appears in both {seen[i]} and {name}")
            seen[i] = name


def freeze_labeled_manifest(
    *,
    n_total: int,
    seed: int,
    n_train: int,
    n_val: int,
    n_test_id: int,
    n_ood_each: int,
    subset_sizes: list[int],
    family_by_index: list[str],
) -> dict[str, Any]:
    """Create a frozen labeled split manifest.

    Ordering assumption: examples are stored as
    [ID pool ...][OOD_narrow...][OOD_strong...][OOD_double...]
    with lengths matching the counts above.
    """
    expected = n_train + n_val + n_test_id + 3 * n_ood_each
    if n_total != expected:
        raise ValueError(f"n_total={n_total} != expected layout size {expected}")
    if len(family_by_index) != n_total:
        raise ValueError("family_by_index length mismatch")

    rng = np.random.default_rng(seed)
    id_count = n_train + n_val + n_test_id
    id_indices = np.arange(id_count, dtype=np.int64)
    perm = rng.permutation(id_indices)
    train_pool = perm[:n_train].tolist()
    validation = perm[n_train : n_train + n_val].tolist()
    test_ID = perm[n_train + n_val :].tolist()

    base = id_count
    test_OOD_narrow = list(range(base, base + n_ood_each))
    base += n_ood_each
    test_OOD_strong = list(range(base, base + n_ood_each))
    base += n_ood_each
    test_OOD_double = list(range(base, base + n_ood_each))

    splits = {
        "train_pool": train_pool,
        "validation": validation,
        "test_ID": test_ID,
        "test_OOD_narrow": test_OOD_narrow,
        "test_OOD_strong": test_OOD_strong,
        "test_OOD_double": test_OOD_double,
    }
    assert_no_split_overlap(splits)
    subsets = build_nested_subsets(train_pool, subset_sizes, seed=seed + 1)

    # Seed uniqueness across ID examples
    return {
        "seed": seed,
        "n_total": n_total,
        "splits": splits,
        "subsets": subsets,
        "family_by_index": family_by_index,
        "layout": {
            "n_train": n_train,
            "n_val": n_val,
            "n_test_id": n_test_id,
            "n_ood_each": n_ood_each,
            "subset_sizes": subset_sizes,
        },
    }
