#!/usr/bin/env python3
"""Fresh confirmatory study for physics-calibrated Krylov-JEPA.

This study is intentionally independent of the historical smoke test examples.
It compares matched physics-aware scratch, legacy Krylov pretraining, and
projected-Ritz Krylov pretraining over label budgets and OOD splits. Hyperparams
are frozen from the bounded calibration stage before generating this corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.datasets import UnlabeledKrylovDataset
from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.baselines import sine_ritz
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.evaluation.metrics import evaluate_example, summarize_metrics
from spectral_krylov_jepa.models.encoders import PotentialEncoder, default_encoder_kwargs
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.io import write_json


LABEL_BUDGETS = [5, 10, 20, 40]
SEEDS = [11, 23, 47]
THRESHOLDS = {
    "fidelity_min": 0.99,
    "residual_true_e_max": 5.0,
    "rel_energy_error_max": 0.06,
}


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finetune_cfg(seed: int) -> dict:
    return {
        "seed": seed,
        "device": "cpu",
        "model_size": "smoke",
        "img_size": 16,
        "batch_size": 4,
        "epochs": 35,
        "lr": 1e-3,
        "early_stopping_patience": 10,
        "num_workers": 0,
        "lambda_psi": 1.0,
        "lambda_e": 1.0,
        "lambda_psi_mse": 0.1,
        "decoder_type": "sine",
        "sine_modes": 25,
        "lambda_residual": 1.0,
        "lambda_rayleigh": 0.1,
        "residual_mode": "scaled",
        "standardize_energy": True,
        "standardize_potential": True,
        "grad_clip": 1.0,
    }


def _pretrain_cfg(seed: int, projected: bool) -> dict:
    cfg = {
        "seed": seed,
        "device": "cpu",
        "model_size": "smoke",
        "img_size": 16,
        "batch_size": 4,
        "max_steps": 120,
        "lr": 3e-4,
        "context_steps": 2,
        "num_workers": 0,
        "log_every": 60,
    }
    if projected:
        cfg.update({"lambda_projected_ritz": 1.0, "projected_modes": 9})
    return cfg


def _encoder_diagnostics(
    encoder_path: str | Path | None,
    data_path: Path,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    kwargs = default_encoder_kwargs("smoke")
    enc = PotentialEncoder(img_size=16, **kwargs)
    if encoder_path is not None:
        payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
        enc.load_state_dict(payload["encoder"], strict=True)
    enc.eval()

    ds = UnlabeledKrylovDataset(data_path, start_index=0)
    zs = []
    with torch.no_grad():
        for i in range(min(len(ds), 120)):
            v = ds[i]["potential"].unsqueeze(0)
            z, _ = enc(v)
            zs.append(z[0].numpy())
    ds.close()

    z = np.asarray(zs, dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(zc, full_matrices=False, compute_uv=False)
    power = s * s
    p = power / max(float(power.sum()), 1e-30)
    effective_rank = float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-30, None)))))
    std = z.std(axis=0)
    cov = np.cov(zc, rowvar=False)
    off = cov - np.diag(np.diag(cov))
    return {
        "effective_rank": effective_rank,
        "min_feature_std": float(std.min()),
        "mean_feature_std": float(std.mean()),
        "offdiag_cov_rms": float(np.sqrt(np.mean(off * off))),
        "feature_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
    }


def _baseline_rows(
    labeled_path: Path,
    manifest: dict,
    grid: GridSpec,
) -> list[dict]:
    ca = cell_area(grid)
    rows: list[dict] = []
    with h5py.File(labeled_path, "r") as f:
        v = np.asarray(f["potential"][:], dtype=np.float64)
        psi = np.asarray(f["psi0"][:], dtype=np.float64)
        energy = np.asarray(f["energy"][:], dtype=np.float64)

    for split, indices in manifest["splits"].items():
        if not split.startswith("test"):
            continue
        for i in indices:
            for name, modes in [("free_box", 1), ("sine_ritz_9", 3)]:
                ehat, phat = sine_ritz(v[i], grid, modes=modes)
                rows.append(
                    {
                        "method": name,
                        "n_labels": 0,
                        "split": split,
                        "index": int(i),
                        **evaluate_example(
                            v[i], psi[i], float(energy[i]), phat, ehat, grid
                        ),
                    }
                )

        for n in LABEL_BUDGETS:
            train = manifest["subsets"][f"n{n}"]
            vt = v[train].reshape(len(train), -1)
            pt = psi[train].reshape(len(train), -1)
            pt *= np.where((pt @ pt[0])[:, None] < 0, -1.0, 1.0)
            y = np.column_stack([pt, energy[train]])
            xmean = vt.mean(axis=0)
            ymean = y.mean(axis=0)
            xc = vt - xmean
            coef = np.linalg.solve(xc @ xc.T + np.eye(len(train)), y - ymean)
            for i in indices:
                pred = (v[i].ravel() - xmean) @ xc.T @ coef + ymean
                phat = pred[:-1]
                phat = phat / np.sqrt(max(float((phat**2).sum() * ca), 1e-30))
                rows.append(
                    {
                        "method": "linear_ridge",
                        "n_labels": n,
                        "split": split,
                        "index": int(i),
                        **evaluate_example(
                            v[i],
                            psi[i],
                            float(energy[i]),
                            phat.reshape(grid.ny, grid.nx),
                            float(pred[-1]),
                            grid,
                        ),
                    }
                )
    return rows


def _summarize_baselines(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["n_labels"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (method, n_labels, split), items in sorted(grouped.items()):
        metrics = [
            {k: v for k, v in r.items() if k not in {"method", "n_labels", "split", "index"}}
            for r in items
        ]
        out.append(
            {
                "method": method,
                "n_labels": n_labels,
                "split": split,
                **summarize_metrics(metrics),
            }
        )
    return out


def _aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for r in rows:
        key = (str(r["method"]), int(r["n_labels"]), str(r["split"]))
        grouped.setdefault(key, []).append(r)
    out = []
    for (method, n_labels, split), items in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "n_labels": n_labels,
                "split": split,
                "n_seeds": len(items),
                "fidelity_mean": float(np.mean([x["fidelity_mean"] for x in items])),
                "fidelity_seed_std": float(np.std([x["fidelity_mean"] for x in items])),
                "residual_true_e_mean": float(
                    np.mean([x["residual_true_e_mean"] for x in items])
                ),
                "residual_true_e_seed_std": float(
                    np.std([x["residual_true_e_mean"] for x in items])
                ),
                "rel_energy_error_mean": float(
                    np.mean([x["rel_energy_error_mean"] for x in items])
                ),
                "elapsed_sec_mean": float(np.mean([x["elapsed_sec"] for x in items])),
            }
        )
    return out


def _first_threshold_budget(aggregate: list[dict], method: str) -> int | None:
    for n in LABEL_BUDGETS:
        matches = [
            r for r in aggregate
            if r["method"] == method and r["n_labels"] == n and r["split"] == "test_ID"
        ]
        if not matches:
            continue
        r = matches[0]
        if (
            r["fidelity_mean"] >= THRESHOLDS["fidelity_min"]
            and r["residual_true_e_mean"] <= THRESHOLDS["residual_true_e_max"]
            and r["rel_energy_error_mean"] <= THRESHOLDS["rel_energy_error_max"]
        ):
            return n
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="confirmatory_artifacts")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grid = GridSpec(n_interior=16)
    unlabeled_path = out / "unlabeled_fresh.h5"
    labeled_path = out / "labeled_fresh.h5"
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    generate_unlabeled_dataset(
        output_path=unlabeled_path,
        n_examples=240,
        n_starts=1,
        depth=3,
        grid=grid,
        base_seed=100_000,
        manifest_path=out / "unlabeled_fresh.manifest.json",
    )
    manifest = generate_labeled_dataset(
        output_path=labeled_path,
        grid=grid,
        n_train=60,
        n_val=20,
        n_test_id=24,
        n_ood_each=16,
        subset_sizes=LABEL_BUDGETS,
        split_seed=92_929,
        id_base_seed=120_000,
        ood_base_seed=140_000,
        manifest_name="confirmatory_labeled_splits",
        manifest_directory=manifest_dir,
    )
    manifest_path = manifest_dir / "confirmatory_labeled_splits.json"

    neural_rows: list[dict] = []
    diagnostics: list[dict] = []

    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        legacy_pre = pretrain(
            method="krylov",
            data_path=unlabeled_path,
            run_dir=seed_dir / "pretrain_krylov",
            config=_pretrain_cfg(seed, projected=False),
        )
        projected_pre = pretrain(
            method="krylov",
            data_path=unlabeled_path,
            run_dir=seed_dir / "pretrain_krylov_projected",
            config=_pretrain_cfg(seed, projected=True),
        )
        encoders = {
            "scratch_physics": None,
            "krylov_physics": legacy_pre["encoder_path"],
            "krylov_projected_physics": projected_pre["encoder_path"],
        }

        for name, encoder in encoders.items():
            diagnostics.append(
                {
                    "seed": seed,
                    "method": name.replace("_physics", ""),
                    **_encoder_diagnostics(encoder, unlabeled_path, seed),
                }
            )

        for n in LABEL_BUDGETS:
            for method, encoder in encoders.items():
                cfg = _finetune_cfg(seed)
                ft = finetune(
                    data_path=labeled_path,
                    manifest_path=manifest_path,
                    subset=f"n{n}",
                    encoder_path=encoder,
                    method_name=method,
                    run_dir=seed_dir / f"finetune_{method}_n{n}",
                    config=cfg,
                )
                for split in [
                    "test_ID",
                    "test_OOD_narrow",
                    "test_OOD_strong",
                    "test_OOD_double",
                ]:
                    if split != "test_ID" and n != max(LABEL_BUDGETS):
                        continue
                    ev = evaluate_checkpoint(
                        checkpoint=ft["checkpoint"],
                        data_path=labeled_path,
                        manifest_path=manifest_path,
                        split=split,
                        config=cfg,
                        output_dir=seed_dir / f"eval_{method}_n{n}_{split}",
                        n_bootstrap=500,
                    )
                    s = ev["summary"]
                    neural_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "n_labels": n,
                            "split": split,
                            "fidelity_mean": s["fidelity_mean"],
                            "rel_energy_error_mean": s["rel_energy_error_mean"],
                            "residual_true_e_mean": s["residual_true_e_mean"],
                            "residual_rayleigh_mean": s["residual_rayleigh_mean"],
                            "elapsed_sec": ft["metrics"]["elapsed_sec"],
                        }
                    )

    aggregate = _aggregate_seed_rows(neural_rows)
    baseline_raw = _baseline_rows(labeled_path, manifest, grid)
    baseline_summary = _summarize_baselines(baseline_raw)

    threshold_budgets = {
        method: _first_threshold_budget(aggregate, method)
        for method in [
            "scratch_physics",
            "krylov_physics",
            "krylov_projected_physics",
        ]
    }

    _write_csv(neural_rows, out / "seed_results.csv")
    _write_csv(aggregate, out / "label_efficiency.csv")
    _write_csv(diagnostics, out / "representation_diagnostics.csv")
    _write_csv(baseline_summary, out / "baseline_summary.csv")

    ood_rows = [r for r in aggregate if r["split"].startswith("test_OOD")]
    _write_csv(ood_rows, out / "ood_results.csv")

    success = {
        "thresholds_frozen_before_fresh_test": THRESHOLDS,
        "first_budget_meeting_joint_threshold": threshold_budgets,
        "krylov_label_efficiency_better_than_scratch": (
            threshold_budgets["krylov_physics"] is not None
            and (
                threshold_budgets["scratch_physics"] is None
                or threshold_budgets["krylov_physics"] < threshold_budgets["scratch_physics"]
            )
        ),
        "projected_krylov_label_efficiency_better_than_scratch": (
            threshold_budgets["krylov_projected_physics"] is not None
            and (
                threshold_budgets["scratch_physics"] is None
                or threshold_budgets["krylov_projected_physics"]
                < threshold_budgets["scratch_physics"]
            )
        ),
    }

    summary = {
        "status": "ok",
        "study_type": "fresh confirmatory bounded study",
        "fresh_seed_ranges": {
            "unlabeled_base_seed": 100_000,
            "id_base_seed": 120_000,
            "ood_base_seed": 140_000,
            "split_seed": 92_929,
        },
        "seeds": SEEDS,
        "label_budgets": LABEL_BUDGETS,
        "thresholds": THRESHOLDS,
        "success": success,
        "aggregate": aggregate,
        "baselines": baseline_summary,
        "representation_diagnostics": diagnostics,
        "elapsed_sec": time.time() - t0,
        "claim_boundary": (
            "This is a fresh finite synthetic 16x16 study with three independent "
            "pretraining/fine-tuning seeds. It can test bounded label efficiency and "
            "OOD behavior on the frozen families, but not universal superiority, "
            "real-molecule validity, or compute-matched superiority to classical Ritz."
        ),
    }
    write_json(summary, out / "summary.json")

    lines = [
        "# Krylov-JEPA fresh confirmatory calibration report",
        "",
        "## Joint physical-accuracy threshold",
        "",
        f"Fidelity >= {THRESHOLDS['fidelity_min']}, residual <= {THRESHOLDS['residual_true_e_max']}, relative energy error <= {THRESHOLDS['rel_energy_error_max']}.",
        "",
        "| Method | First label budget meeting threshold |",
        "|---|---:|",
    ]
    for method, budget in threshold_budgets.items():
        lines.append(f"| {method} | {budget if budget is not None else 'none'} |")
    lines += [
        "",
        "## ID label-efficiency aggregate",
        "",
        "| Method | Labels | Fidelity | Residual @ true E | Relative energy error |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        if r["split"] == "test_ID":
            lines.append(
                f"| {r['method']} | {r['n_labels']} | {r['fidelity_mean']:.6f} | "
                f"{r['residual_true_e_mean']:.6f} | {r['rel_energy_error_mean']:.6f} |"
            )
    lines += [
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    (out / "KRYLOV_CALIBRATION_REPORT.md").write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
