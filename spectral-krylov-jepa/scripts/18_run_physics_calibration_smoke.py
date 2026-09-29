#!/usr/bin/env python3
"""Run a bounded physics-calibration smoke study.

This is deliberately small enough for CI. It tests whether the physics-aware
objective and sine decoder move the failure mode in the right direction. It is
not the final confirmatory experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.io import write_json


FROZEN_REFERENCE = {
    "krylov_n20_fidelity": 0.9916496889541183,
    "krylov_n20_residual_true_e": 40.19089036780306,
    "scratch_n20_fidelity": 0.9939449302733622,
    "scratch_n20_residual_true_e": 29.04363463159122,
    "free_box_fidelity": 0.994460,
    "free_box_residual_true_e": 1.483805,
    "ritz9_fidelity": 0.999898,
    "ritz9_residual_true_e": 0.761666,
    "ridge_fidelity": 0.999904,
    "ridge_residual_true_e": 0.516766,
}


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _base_finetune_config(seed: int) -> dict:
    return {
        "seed": seed,
        "device": "cpu",
        "model_size": "smoke",
        "img_size": 16,
        "batch_size": 4,
        "epochs": 25,
        "lr": 1e-3,
        "early_stopping_patience": 8,
        "num_workers": 0,
        "lambda_psi": 1.0,
        "lambda_e": 1.0,
        "lambda_psi_mse": 0.1,
        "standardize_energy": True,
        "standardize_potential": True,
        "grad_clip": 1.0,
    }


def _variant_config(seed: int, name: str) -> dict:
    cfg = _base_finetune_config(seed)
    if name in {"scratch_legacy", "krylov_legacy"}:
        cfg.update(
            {
                "decoder_type": "pixel",
                "lambda_residual": 0.0,
                "lambda_rayleigh": 0.0,
            }
        )
    elif name in {"scratch_physics", "krylov_physics"}:
        cfg.update(
            {
                "decoder_type": "sine",
                "sine_modes": 25,
                "lambda_residual": 1.0,
                "lambda_rayleigh": 0.1,
                "residual_mode": "scaled",
            }
        )
    elif name == "krylov_residual_pixel":
        cfg.update(
            {
                "decoder_type": "pixel",
                "lambda_residual": 1.0,
                "lambda_rayleigh": 0.1,
                "residual_mode": "scaled",
            }
        )
    elif name == "krylov_sine_no_physics":
        cfg.update(
            {
                "decoder_type": "sine",
                "sine_modes": 25,
                "lambda_residual": 0.0,
                "lambda_rayleigh": 0.0,
            }
        )
    else:
        raise ValueError(f"Unknown variant {name}")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="calibration_artifacts")
    parser.add_argument("--seeds", default="11,23,47")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 1:
        raise ValueError("At least one seed is required")

    grid = GridSpec(n_interior=16)
    unlabeled_path = out / "unlabeled.h5"
    labeled_path = out / "labeled.h5"
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    generate_unlabeled_dataset(
        output_path=unlabeled_path,
        n_examples=120,
        n_starts=1,
        depth=3,
        grid=grid,
        base_seed=10_000,
        manifest_path=out / "unlabeled.manifest.json",
    )
    generate_labeled_dataset(
        output_path=labeled_path,
        grid=grid,
        n_train=40,
        n_val=12,
        n_test_id=12,
        n_ood_each=8,
        subset_sizes=[20],
        split_seed=2026,
        manifest_name="calibration_labeled_splits",
        manifest_directory=manifest_dir,
    )
    manifest_path = manifest_dir / "calibration_labeled_splits.json"

    rows: list[dict] = []
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pretrain_cfg = {
            "seed": seed,
            "device": "cpu",
            "model_size": "smoke",
            "img_size": 16,
            "batch_size": 4,
            "max_steps": 80,
            "lr": 3e-4,
            "context_steps": 2,
            "num_workers": 0,
            "log_every": 40,
        }
        pre = pretrain(
            method="krylov",
            data_path=unlabeled_path,
            run_dir=seed_dir / "pretrain_krylov",
            config=pretrain_cfg,
        )
        encoder_path = pre["encoder_path"]

        variants = [
            "scratch_legacy",
            "krylov_legacy",
            "scratch_physics",
            "krylov_physics",
        ]
        if seed == seeds[0]:
            variants.extend(["krylov_residual_pixel", "krylov_sine_no_physics"])

        seed_result: dict[str, dict] = {}
        for variant in variants:
            cfg = _variant_config(seed, variant)
            encoder = encoder_path if variant.startswith("krylov_") else None
            ft = finetune(
                data_path=labeled_path,
                manifest_path=manifest_path,
                subset="n20",
                encoder_path=encoder,
                method_name=variant,
                run_dir=seed_dir / f"finetune_{variant}",
                config=cfg,
            )
            ev = evaluate_checkpoint(
                checkpoint=ft["checkpoint"],
                data_path=labeled_path,
                manifest_path=manifest_path,
                split="test_ID",
                config=cfg,
                output_dir=seed_dir / f"eval_{variant}",
                n_bootstrap=300,
            )
            s = ev["summary"]
            row = {
                "seed": seed,
                "variant": variant,
                "fidelity_mean": s["fidelity_mean"],
                "rel_energy_error_mean": s["rel_energy_error_mean"],
                "residual_true_e_mean": s["residual_true_e_mean"],
                "residual_rayleigh_mean": s["residual_rayleigh_mean"],
                "residual_predicted_e_mean": s["residual_rel_mean"],
            }
            rows.append(row)
            seed_result[variant] = row
        per_seed[str(seed)] = seed_result

    _write_csv(rows, out / "calibration_results.csv")

    principal = [r for r in rows if r["variant"] in {
        "scratch_legacy",
        "krylov_legacy",
        "scratch_physics",
        "krylov_physics",
    }]
    aggregate = {}
    for variant in sorted({r["variant"] for r in principal}):
        vr = [r for r in principal if r["variant"] == variant]
        aggregate[variant] = {
            "n_seeds": len(vr),
            "fidelity_mean_across_seeds": float(np.mean([r["fidelity_mean"] for r in vr])),
            "fidelity_std_across_seeds": float(np.std([r["fidelity_mean"] for r in vr])),
            "residual_true_e_mean_across_seeds": float(
                np.mean([r["residual_true_e_mean"] for r in vr])
            ),
            "residual_true_e_std_across_seeds": float(
                np.std([r["residual_true_e_mean"] for r in vr])
            ),
            "rel_energy_error_mean_across_seeds": float(
                np.mean([r["rel_energy_error_mean"] for r in vr])
            ),
        }

    kp = aggregate.get("krylov_physics", {})
    kl = aggregate.get("krylov_legacy", {})
    sp = aggregate.get("scratch_physics", {})
    levels = {
        "level_1_residual_improves_vs_original": (
            kp.get("residual_true_e_mean_across_seeds", float("inf"))
            < kl.get("residual_true_e_mean_across_seeds", float("-inf"))
        ),
        "level_2_beats_matched_scratch_on_residual": (
            kp.get("residual_true_e_mean_across_seeds", float("inf"))
            < sp.get("residual_true_e_mean_across_seeds", float("-inf"))
        ),
        "level_3_label_efficiency": None,
        "level_4_ood": None,
        "level_5_beats_ritz_or_ridge": None,
        "level_6_compute_controlled": None,
    }

    summary = {
        "status": "ok",
        "study_type": "bounded calibration smoke; not confirmatory",
        "seeds": seeds,
        "frozen_reference": FROZEN_REFERENCE,
        "aggregate": aggregate,
        "success_levels": levels,
        "per_seed": per_seed,
        "claim_boundary": (
            "This CI study tests whether the proposed downstream calibration changes the "
            "known failure mode. It does not establish state of the art, full label "
            "efficiency, OOD robustness, or compute-matched superiority."
        ),
    }
    write_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
