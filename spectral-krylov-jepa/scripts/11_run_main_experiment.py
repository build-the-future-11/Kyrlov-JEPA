#!/usr/bin/env python3
"""First decisive label-efficiency experiment (long-running).

Compares Scratch / Field-JEPA / Operator-JEPA / Krylov-JEPA on frozen protocol
subsets N∈{10,25,50,100}, seeds {11,23,47}, evaluated on test_ID and every OOD split.
Downstream loss weights follow `paper/experiment_protocol.md` (λ_ψ=1, λ_E=1, λ_ψmse=0.1).

Resumable: pass --resume-run <dir name under experiments/raw> to skip completed
pretrain / fine-tune / eval steps.

Example:
  python scripts/11_run_main_experiment.py --config configs/smoke/main_experiment.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.plotting.learning_curves import plot_label_efficiency
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.config import load_config
from spectral_krylov_jepa.utils.provenance import freeze_or_check, source_hashes, sha256, exclusive_run
from spectral_krylov_jepa.utils.io import make_run_dir, write_json
from spectral_krylov_jepa.utils.logging import setup_logger

EVAL_SPLITS = ["test_ID", "test_OOD_narrow", "test_OOD_strong", "test_OOD_double"]
SUMMARY_KEYS = [
    "fidelity_mean",
    "fidelity_std",
    "rel_energy_error_mean",
    "rel_rayleigh_energy_error_mean",
    "residual_rel_mean",
    "residual_true_e_mean",
    "residual_rayleigh_mean",
]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/smoke/main_experiment.yaml"))
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated, e.g. 11,23,47")
    parser.add_argument("--resume-run", type=str, default=None, help="Run dir name under experiments/raw")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="label_efficiency",
        help="Stem for results/tables/<prefix>*.csv and figures/<prefix>.png",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg.get("seeds", [11]))]
    methods = cfg.get("methods", ["scratch", "field", "operator", "krylov"])
    subsets = [f"n{n}" for n in cfg.get("subset_sizes", [10, 25, 50, 100])]
    manifest_name = str(cfg.get("manifest_name", "main_labeled_splits"))

    raw_root = ROOT / "experiments" / "raw"
    if args.resume_run:
        run_root = raw_root / args.resume_run
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
        if not (run_root / "identity.json").exists():
            raise ValueError("Legacy run has no source identity; preserve it and start a new run")
    else:
        run_root = make_run_dir(raw_root, str(cfg.get("run_name", "main_label_efficiency")))
    log = setup_logger("main_exp", log_file=run_root / "run.log")
    grid = GridSpec(n_interior=int(cfg.get("n_interior", 32)))
    freeze_or_check(run_root / "identity.json", {"config": cfg, "seeds": seeds, "sources": source_hashes(ROOT)})
    if not (run_root / "run_config.json").exists():
        write_json({"config": cfg, "seeds": seeds, "git_sha": _git_sha()}, run_root / "run_config.json")

    unlab = run_root / "unlabeled.h5"
    labeled = run_root / "labeled.h5"
    manifest_path = run_root / f"{manifest_name}.json"
    data_done = run_root / "data.done"
    if not data_done.exists():
        log.info("Generating unlabeled (%d)...", int(cfg["n_unlabeled"]))
        generate_unlabeled_dataset(
            output_path=unlab,
            n_examples=int(cfg["n_unlabeled"]),
            n_starts=int(cfg.get("n_starts", 2)),
            depth=int(cfg.get("depth", 3)),
            grid=grid,
            base_seed=10_000,
            manifest_path=run_root / "unlabeled.manifest.json",
        )
        log.info("Generating labeled...")
        generate_labeled_dataset(
            output_path=labeled,
            grid=grid,
            n_train=int(cfg["n_train"]),
            n_val=int(cfg["n_val"]),
            n_test_id=int(cfg["n_test_id"]),
            n_ood_each=int(cfg["n_ood_each"]),
            subset_sizes=list(cfg["subset_sizes"]),
            split_seed=2026,
            manifest_name=manifest_name,
            manifest_directory=run_root,
        )
        data_done.write_text("ok\n")
    else:
        log.info("Resume: reusing data in %s", run_root)

    freeze_or_check(run_root / "data_hashes.json", {p.name: sha256(p) for p in [unlab, labeled, manifest_path]})

    pretrain_cfg = {
        "device": cfg.get("device", "auto"),
        "model_size": cfg.get("model_size", "default"),
        "img_size": grid.n_interior,
        "batch_size": int(cfg.get("batch_size", 8)),
        "max_steps": int(cfg.get("pretrain_steps", 2000)),
        "lr": 3e-4,
        "context_steps": int(cfg.get("context_steps", 2)),
        "num_workers": 0,
    }
    finetune_cfg_base = {
        "device": cfg.get("device", "auto"),
        "model_size": cfg.get("model_size", "default"),
        "img_size": grid.n_interior,
        "batch_size": int(cfg.get("batch_size", 8)),
        "epochs": int(cfg.get("finetune_epochs", 80)),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 15)),
        "num_workers": 0,
        "lambda_psi": float(cfg.get("lambda_psi", 1.0)),
        "lambda_e": float(cfg.get("lambda_e", 1.0)),
        "lambda_psi_mse": float(cfg.get("lambda_psi_mse", 0.1)),
        "standardize_energy": True,
        "standardize_potential": True,
    }
    n_bootstrap = int(cfg.get("n_bootstrap", 2000))

    # Pretrain once per method (seed 0 for SSL), then fine-tune over seeds/subsets
    encoders: dict[str, str | None] = {"scratch": None}
    for method in methods:
        if method == "scratch":
            continue
        pre_dir = run_root / f"pretrain_{method}"
        if (pre_dir / "encoder.pt").exists() and (pre_dir / "metrics.json").exists():
            log.info("Resume: pretrain %s complete", method)
            encoders[method] = str(pre_dir / "encoder.pt")
            continue
        log.info("Pretraining %s", method)
        pre = pretrain(
            method=method,  # type: ignore[arg-type]
            data_path=unlab,
            run_dir=pre_dir,
            config=dict(pretrain_cfg, seed=0),
        )
        encoders[method] = pre["encoder_path"]

    rows = []
    for method in methods:
        for seed in seeds:
            for subset in subsets:
                ft_cfg = dict(finetune_cfg_base, seed=seed)
                ft_dir = run_root / f"ft_{method}_{subset}_s{seed}"
                if (ft_dir / "checkpoint_best.pt").exists() and (ft_dir / "metrics.json").exists():
                    log.info("Resume: finetune %s %s seed=%d complete", method, subset, seed)
                else:
                    log.info("Finetune %s %s seed=%d", method, subset, seed)
                    finetune(
                        data_path=labeled,
                        manifest_path=manifest_path,
                        subset=subset,
                        encoder_path=encoders[method],
                        method_name=method,
                        run_dir=ft_dir,
                        config=ft_cfg,
                    )
                for split in EVAL_SPLITS:
                    ev_dir = run_root / f"eval_{method}_{subset}_s{seed}" / split
                    ev_json = ev_dir / "summary.json"
                    if ev_json.exists():
                        summary = json.loads(ev_json.read_text())
                    else:
                        ev = evaluate_checkpoint(
                            checkpoint=ft_dir / "checkpoint_best.pt",
                            data_path=labeled,
                            manifest_path=manifest_path,
                            split=split,
                            config=ft_cfg,
                            output_dir=ev_dir,
                            n_bootstrap=n_bootstrap,
                        )
                        summary = dict(ev["summary"])
                        boot = ev.get("bootstrap_fidelity") or {}
                        summary["bootstrap_fidelity_ci_low"] = boot.get("ci_low", float("nan"))
                        summary["bootstrap_fidelity_ci_high"] = boot.get("ci_high", float("nan"))
                        write_json(summary, ev_json)
                    row = {"method": method, "n_labels": int(subset[1:]), "seed": seed, "split": split}
                    for k in SUMMARY_KEYS + ["bootstrap_fidelity_ci_low", "bootstrap_fidelity_ci_high"]:
                        row[k] = summary.get(k, float("nan"))
                    rows.append(row)

    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_seed = tables / f"{args.output_prefix}_per_seed.csv"
    with per_seed.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    grouped: dict[tuple, list] = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["method"], r["n_labels"])].append(r)
    agg = []
    for (split, method, n_labels), rs in sorted(grouped.items()):
        fids = [r["fidelity_mean"] for r in rs]
        agg.append(
            {
                "split": split,
                "method": method,
                "n_labels": n_labels,
                "fidelity_mean": float(np.mean(fids)),
                "fidelity_std": float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0,
                "rel_energy_error_mean": float(np.mean([r["rel_energy_error_mean"] for r in rs])),
                "rel_rayleigh_energy_error_mean": float(
                    np.mean([r["rel_rayleigh_energy_error_mean"] for r in rs])
                ),
                "residual_rel_mean": float(np.mean([r["residual_rel_mean"] for r in rs])),
                "n_seeds": len(rs),
            }
        )
    table = tables / f"{args.output_prefix}.csv"
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    fig_dir = ROOT / "figures"
    plot_label_efficiency(
        [a for a in agg if a["split"] == "test_ID"],
        fig_dir / f"{args.output_prefix}.png",
        title="Label efficiency (test_ID; mean ± std over seeds)",
    )
    for split in EVAL_SPLITS[1:]:
        plot_label_efficiency(
            [a for a in agg if a["split"] == split],
            fig_dir / f"{args.output_prefix}_{split}.png",
            title=f"Label efficiency ({split}; mean ± std over seeds)",
        )
    write_json(
        {"rows": rows, "aggregate": agg, "run_dir": str(run_root), "git_sha": _git_sha()},
        run_root / "summary.json",
    )
    log.info("Wrote %s, %s and figures/%s*.png", table, per_seed, args.output_prefix)
    return 0


if __name__ == "__main__":
    with exclusive_run(ROOT / "experiments/raw/main_runner.lock"):
        raise SystemExit(main())
