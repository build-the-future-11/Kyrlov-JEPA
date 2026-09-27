#!/usr/bin/env python3
"""Run a named ablation (pretrain + finetune + eval)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.ablations import get_ablation_config, list_ablations
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.config import load_config
from spectral_krylov_jepa.utils.io import make_run_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True, choices=list_ablations())
    parser.add_argument("--config", type=str, default="configs/smoke/ablation.yaml")
    parser.add_argument("--labeled", type=str, default="experiments/raw/labeled.h5")
    parser.add_argument("--manifest", type=str, default="experiments/manifests/labeled_splits.json")
    parser.add_argument("--subset", type=str, default="n10")
    args = parser.parse_args()

    base = load_config(args.config) if Path(args.config).exists() else {}
    abl = get_ablation_config(args.ablation)
    cfg = {**base, **abl}
    method = cfg["method"]
    run_root = make_run_dir("experiments/raw", f"ablation_{args.ablation}")

    # Optionally generate shuffled unlabeled data
    unlab = run_root / "unlabeled.h5"
    generate_unlabeled_dataset(
        output_path=unlab,
        n_examples=int(cfg.get("n_unlabeled", 100)),
        n_starts=int(cfg.get("n_starts", 1)),
        depth=int(cfg.get("depth", 3)),
        grid=GridSpec(n_interior=int(cfg.get("img_size", 32))),
        base_seed=int(cfg.get("base_seed", 50_000)),
        shuffle_physics=bool(cfg.get("shuffle_physics", False)),
    )

    pre = pretrain(
        method=method,
        data_path=unlab,
        run_dir=run_root / "pretrain",
        config=cfg,
    )
    ft = finetune(
        data_path=args.labeled,
        manifest_path=args.manifest,
        subset=args.subset,
        encoder_path=pre["encoder_path"],
        method_name=args.ablation,
        run_dir=run_root / "finetune",
        config=cfg,
    )
    ev = evaluate_checkpoint(
        checkpoint=Path(ft["run_dir"]) / "checkpoint_best.pt",
        data_path=args.labeled,
        manifest_path=args.manifest,
        split="test_ID",
        config=cfg,
        output_dir=run_root / "eval",
    )
    write_json({"ablation": args.ablation, "pretrain": pre, "finetune": ft, "eval": ev}, run_root / "summary.json")
    print(ev["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
