"""Tiny end-to-end smoke test (generation → short train → eval)."""

from pathlib import Path

import torch

from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain


def test_end_to_end_smoke(tmp_path: Path):
    g = GridSpec(n_interior=8)
    unlab = tmp_path / "u.h5"
    generate_unlabeled_dataset(
        output_path=unlab,
        n_examples=8,
        n_starts=1,
        depth=2,
        grid=g,
        base_seed=42,
    )
    lab = tmp_path / "l.h5"
    manifest = generate_labeled_dataset(
        output_path=lab,
        grid=g,
        n_train=6,
        n_val=2,
        n_test_id=2,
        n_ood_each=2,
        subset_sizes=[4],
        split_seed=0,
        manifest_name="e2e_test",
        manifest_directory=tmp_path,
        id_base_seed=9000,
        ood_base_seed=9100,
    )
    # manifest written to experiments/manifests — also local copy
    man_path = tmp_path / "manifest.json"
    import json

    man_path.write_text(json.dumps(manifest))

    cfg_pre = {
        "seed": 0,
        "device": "cpu",
        "model_size": "smoke",
        "img_size": 8,
        "batch_size": 2,
        "max_steps": 4,
        "context_steps": 1,
        "num_workers": 0,
        "log_every": 1,
    }
    # depth=2 gives q0,q1,q2 — context_steps=1 predicts q1
    pre = pretrain(method="krylov", data_path=unlab, run_dir=tmp_path / "pre", config=cfg_pre)
    assert Path(pre["encoder_path"]).exists()

    cfg_ft = {
        "seed": 0,
        "device": "cpu",
        "model_size": "smoke",
        "img_size": 8,
        "batch_size": 2,
        "epochs": 2,
        "early_stopping_patience": 5,
        "num_workers": 0,
    }
    ft = finetune(
        data_path=lab,
        manifest_path=man_path,
        subset="n4",
        encoder_path=pre["encoder_path"],
        method_name="krylov",
        run_dir=tmp_path / "ft",
        config=cfg_ft,
    )
    ev = evaluate_checkpoint(
        checkpoint=Path(ft["run_dir"]) / "checkpoint_best.pt",
        data_path=lab,
        manifest_path=man_path,
        split="test_ID",
        config=cfg_ft,
        output_dir=tmp_path / "eval",
        n_bootstrap=50,
    )
    assert "fidelity_mean" in ev["summary"]
    assert Path(tmp_path / "eval" / "metrics.json").exists()
