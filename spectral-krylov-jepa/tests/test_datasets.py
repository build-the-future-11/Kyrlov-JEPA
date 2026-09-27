"""Dataset and split tests."""

from pathlib import Path

import numpy as np

from spectral_krylov_jepa.data.datasets import LabeledEigenstateDataset, UnlabeledKrylovDataset
from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.data.splits import assert_no_split_overlap, build_nested_subsets
from spectral_krylov_jepa.physics.grid import GridSpec


def test_nested_subsets():
    subsets = build_nested_subsets(list(range(100)), [10, 25, 50], seed=0)
    assert set(subsets["n10"]).issubset(set(subsets["n25"]))
    assert set(subsets["n25"]).issubset(set(subsets["n50"]))


def test_unlabeled_and_labeled_generation(tmp_path: Path):
    g = GridSpec(n_interior=8)
    unlab = tmp_path / "u.h5"
    generate_unlabeled_dataset(
        output_path=unlab,
        n_examples=4,
        n_starts=1,
        depth=2,
        grid=g,
        base_seed=100,
    )
    ds = UnlabeledKrylovDataset(unlab)
    assert len(ds) == 4
    item = ds[0]
    assert item["potential"].shape == (8, 8)
    assert item["q"].shape == (3, 64)
    ds.close()

    lab = tmp_path / "l.h5"
    manifest = generate_labeled_dataset(
        output_path=lab,
        grid=g,
        n_train=6,
        n_val=2,
        n_test_id=2,
        n_ood_each=2,
        subset_sizes=[2, 4],
        split_seed=1,
        manifest_name="test_splits",
        manifest_directory=tmp_path,
        id_base_seed=2000,
        ood_base_seed=3000,
    )
    assert_no_split_overlap(manifest["splits"])
    lds = LabeledEigenstateDataset(lab, indices=manifest["splits"]["train_pool"])
    assert len(lds) == 6
    row = lds[0]
    assert row["psi0"].shape == (8, 8)
    assert np.isfinite(row["energy"].item())
    lds.close()
