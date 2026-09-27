#!/usr/bin/env python3
"""End-to-end smoke suite: data → pretrain → finetune → evaluate → tables/figures.

This is a proof-of-pipeline, not a claim of scientific success.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.evaluation.ood import evaluate_ood_suite
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state_from_potential
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.physics.validation import assert_physics_ok
from spectral_krylov_jepa.plotting.learning_curves import plot_label_efficiency
from spectral_krylov_jepa.plotting.ood_figures import plot_ood_comparison
from spectral_krylov_jepa.plotting.physics_figures import plot_physics_panel, plot_prediction_panel
from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.training.checkpointing import load_checkpoint
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.config import load_config, save_yaml
from spectral_krylov_jepa.utils.io import ensure_dir, make_run_dir, write_json
from spectral_krylov_jepa.utils.logging import setup_logger


def _write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end smoke suite")
    parser.add_argument("--config", type=str, default="configs/smoke/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    defaults = {
        "n_interior": 16,
        "n_unlabeled": 120,
        "n_starts": 1,
        "depth": 3,
        "n_train": 40,
        "n_val": 12,
        "n_test_id": 12,
        "n_ood_each": 8,
        "subset_sizes": [10, 20],
        "model_size": "smoke",
        "pretrain_steps": 80,
        "finetune_epochs": 25,
        "batch_size": 4,
        "seed": 11,
        "device": "auto",
        "context_steps": 2,
        "include_baselines": True,
        "lambda_psi": 1.0,
        "lambda_e": 1.0,
        "lambda_psi_mse": 0.1,
        "standardize_energy": True,
        "standardize_potential": True,
        "early_stopping_patience": 10,
        "run_ood": True,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    run_root = make_run_dir("experiments/raw", "smoke_suite")
    log = setup_logger("smoke", log_file=run_root / "smoke.log")
    save_yaml(cfg, run_root / "smoke_config.yaml")
    t0 = time.time()
    summary: dict = {"status": "running", "steps": {}}

    grid = GridSpec(n_interior=int(cfg["n_interior"]))
    log.info("Step 0: physics validation")
    report = assert_physics_ok(grid=grid, potential_seed=int(cfg["seed"]))
    summary["steps"]["physics"] = {"ok": report.ok, "energy": report.energy}

    v, _ = generate_potential("id_gaussian_mixture", int(cfg["seed"]), grid=grid)
    gs = solve_ground_state_from_potential(v, grid)
    fig_dir = ensure_dir("figures")
    plot_physics_panel(v, gs.wavefunction, gs.energy, grid, fig_dir / "figure_A_physics_panel.png")

    log.info("Step 1: unlabeled data")
    unlab_path = run_root / "unlabeled.h5"
    generate_unlabeled_dataset(
        output_path=unlab_path,
        n_examples=int(cfg["n_unlabeled"]),
        n_starts=int(cfg["n_starts"]),
        depth=int(cfg["depth"]),
        grid=grid,
        base_seed=10_000,
        manifest_path=run_root / "unlabeled.manifest.json",
    )

    log.info("Step 2: labeled data")
    labeled_path = run_root / "labeled.h5"
    subset_sizes = [s for s in cfg["subset_sizes"] if s <= int(cfg["n_train"])]
    if not subset_sizes:
        subset_sizes = [min(10, int(cfg["n_train"]))]
    manifest = generate_labeled_dataset(
        output_path=labeled_path,
        grid=grid,
        n_train=int(cfg["n_train"]),
        n_val=int(cfg["n_val"]),
        n_test_id=int(cfg["n_test_id"]),
        n_ood_each=int(cfg["n_ood_each"]),
        subset_sizes=subset_sizes,
        split_seed=2026,
        manifest_name="smoke_labeled_splits",
    )
    manifest_path = ROOT / "experiments" / "manifests" / "smoke_labeled_splits.json"

    pretrain_cfg = {
        "seed": int(cfg["seed"]),
        "device": cfg["device"],
        "model_size": cfg["model_size"],
        "img_size": int(cfg["n_interior"]),
        "batch_size": int(cfg["batch_size"]),
        "max_steps": int(cfg["pretrain_steps"]),
        "lr": 3e-4,
        "context_steps": int(cfg["context_steps"]),
        "num_workers": 0,
        "log_every": 20,
    }
    finetune_cfg = {
        "seed": int(cfg["seed"]),
        "device": cfg["device"],
        "model_size": cfg["model_size"],
        "img_size": int(cfg["n_interior"]),
        "batch_size": int(cfg["batch_size"]),
        "epochs": int(cfg["finetune_epochs"]),
        "lr": 1e-3,
        "early_stopping_patience": int(cfg["early_stopping_patience"]),
        "num_workers": 0,
        "lambda_psi": float(cfg["lambda_psi"]),
        "lambda_e": float(cfg["lambda_e"]),
        "lambda_psi_mse": float(cfg["lambda_psi_mse"]),
        "standardize_energy": bool(cfg["standardize_energy"]),
        "standardize_potential": bool(cfg["standardize_potential"]),
    }

    methods_to_run = [("scratch", None)]
    if cfg.get("include_baselines", True):
        methods_to_run.extend([("field", "field"), ("operator", "operator")])
    methods_to_run.append(("krylov", "krylov"))

    # Pretrain each SSL method once
    encoders: dict[str, str | None] = {"scratch": None}
    for method_name, pretrain_method in methods_to_run:
        if pretrain_method is None:
            continue
        log.info("Pretrain: %s", method_name)
        pre = pretrain(
            method=pretrain_method,
            data_path=unlab_path,
            run_dir=run_root / f"pretrain_{method_name}",
            config=pretrain_cfg,
        )
        encoders[method_name] = pre["encoder_path"]
        summary["steps"][f"pretrain_{method_name}"] = pre["metrics"]

    table_rows: list[dict] = []
    best_ckpts: dict[str, Path] = {}

    for method_name, _ in methods_to_run:
        for n_lab in subset_sizes:
            subset = f"n{n_lab}"
            log.info("Finetune: %s %s", method_name, subset)
            ft = finetune(
                data_path=labeled_path,
                manifest_path=manifest_path,
                subset=subset,
                encoder_path=encoders.get(method_name),
                method_name=method_name,
                run_dir=run_root / f"finetune_{method_name}_{subset}",
                config=finetune_cfg,
            )
            ckpt = Path(ft["run_dir"]) / "checkpoint_best.pt"
            if n_lab == max(subset_sizes):
                best_ckpts[method_name] = ckpt

            ev = evaluate_checkpoint(
                checkpoint=ckpt,
                data_path=labeled_path,
                manifest_path=manifest_path,
                split="test_ID",
                config=finetune_cfg,
                output_dir=run_root / f"eval_{method_name}_{subset}",
                n_bootstrap=500,
            )
            s = ev["summary"]
            row = {
                "method": method_name,
                "n_labels": n_lab,
                "seed": int(cfg["seed"]),
                "fidelity_mean": s["fidelity_mean"],
                "fidelity_std": s["fidelity_std"],
                "rel_energy_error_mean": s["rel_energy_error_mean"],
                "rel_rayleigh_energy_error_mean": s.get("rel_rayleigh_energy_error_mean", float("nan")),
                "residual_rel_mean": s["residual_rel_mean"],
                "residual_true_e_mean": s.get("residual_true_e_mean", float("nan")),
                "residual_rayleigh_mean": s.get("residual_rayleigh_mean", float("nan")),
                "run_dir": str(ft["run_dir"]),
            }
            table_rows.append(row)
            summary["steps"][f"eval_{method_name}_{subset}"] = {
                k: row[k]
                for k in row
                if k not in ("method", "n_labels", "seed", "run_dir")
            }

            # Prediction panel for largest subset only
            if n_lab == max(subset_sizes):
                model = DownstreamGroundStateModel(
                    img_size=grid.n_interior,
                    size=cfg["model_size"],
                    cell_area=cell_area(grid),
                )
                load_checkpoint(ckpt, model)
                model.eval()
                with h5py.File(labeled_path, "r") as f:
                    idx = manifest["splits"]["test_ID"][0]
                    vv = np.asarray(f["potential"][idx])
                    psi_t = np.asarray(f["psi0"][idx])
                with torch.no_grad():
                    out = model(torch.from_numpy(vv).unsqueeze(0).float())
                    psi_p = out.psi.numpy()[0]
                plot_prediction_panel(
                    psi_t,
                    psi_p,
                    fig_dir / f"figure_E_prediction_{method_name}.png",
                    fidelity=float(ev["per_example"][0]["fidelity"]),
                )

    # OOD suite on Krylov at largest N (smoke-scale)
    if cfg.get("run_ood", True) and "krylov" in best_ckpts:
        log.info("OOD suite for Krylov")
        ood = evaluate_ood_suite(
            checkpoint=str(best_ckpts["krylov"]),
            data_path=str(labeled_path),
            manifest_path=str(manifest_path),
            config=finetune_cfg,
            output_dir=str(run_root / "ood_krylov"),
        )
        ood_compact = {
            method: {
                split: {
                    "fidelity_mean": res["summary"]["fidelity_mean"],
                    "rel_energy_error_mean": res["summary"]["rel_energy_error_mean"],
                    "residual_true_e_mean": res["summary"].get("residual_true_e_mean"),
                }
                for split, res in splits.items()
            }
            for method, splits in {"krylov": ood}.items()
        }
        # reshape for plot_ood_comparison: method -> split -> result-with-summary
        plot_data = {"krylov": ood}
        plot_ood_comparison(plot_data, fig_dir / "ood_comparison_smoke.png")
        write_json(ood_compact, run_root / "ood_summary.json")
        summary["steps"]["ood_krylov"] = ood_compact["krylov"]

    table_path = ensure_dir("results/tables") / "smoke_label_efficiency.csv"
    _write_table(table_rows, table_path)
    _write_table(table_rows, ensure_dir("results/tables") / "smoke_metrics.csv")
    plot_label_efficiency(
        table_rows,
        fig_dir / "label_efficiency_smoke.png",
        title="Smoke label efficiency (all methods)",
    )

    summary["status"] = "ok"
    summary["elapsed_sec"] = time.time() - t0
    summary["table"] = str(table_path)
    summary["note"] = (
        "Improved smoke with Field/Operator/Krylov/Scratch, energy standardization, "
        "dual residuals, and Rayleigh energy. Not a claim of scientific success. "
        "Full decisive experiment not yet executed."
    )
    write_json(summary, run_root / "summary.json")
    write_json(summary, ensure_dir("experiments/summaries") / "smoke_latest.json")
    write_json(
        {
            "completed_runs": [str(run_root)],
            "smoke_verified": True,
            "full_results_ready": False,
        },
        ROOT / "paper" / "results_status_runtime.json",
    )
    log.info("SMOKE SUITE COMPLETE in %.1fs", summary["elapsed_sec"])
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
