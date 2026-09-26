#!/usr/bin/env python3
"""End-to-end Free-Physics JEPA runner (Study-2 / amendment 0003).

Smoke metrics are NOT scientific evidence.
Study-1 runs under data/{mode}/ are archived negatives; this runner uses data/{mode}_v2/.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "lmop_jepa"))
sys.path.insert(0, str(PKG / "scripts"))

_PROTO = PKG / "protocol.yaml"
if not _PROTO.exists():
    raise FileNotFoundError(f"protocol missing: {_PROTO}")

from darcy_reference import (  # noqa: E402
    DarcyGrid,
    build_darcy_matrix,
    condition_estimate,
    is_spd_probe,
    manufactured_round_trip,
    manufactured_sine_mode,
    matrix_symmetry_error,
    sample_log_permeability,
)
from lmop_jepa.data import (  # noqa: E402
    build_split_manifest,
    generate_genuine_dataset,
    generate_manufactured_dataset,
    write_json,
)
from lmop_jepa.training import (  # noqa: E402
    evaluate_model,
    finetune_genuine,
    get_device,
    set_threads,
    train_lmop,
    train_mml,
)
from aggregate_claim_gate import write_claim_artifacts  # noqa: E402


def physics_validation_report(grid: DarcyGrid, out: Path) -> dict:
    a = sample_log_permeability(grid, 0)
    A = build_darcy_matrix(a, grid)
    u = manufactured_sine_mode(grid, 1, 2, 0.3)
    rt = manufactured_round_trip(a, u, grid, rtol=1e-8)
    report = {
        "grid": grid.to_dict(),
        "symmetry_error": matrix_symmetry_error(A),
        "spd": is_spd_probe(A),
        "condition_est": condition_estimate(A),
        "roundtrip": rt,
        "ok": bool(rt["ok"] and matrix_symmetry_error(A) < 1e-10 and is_spd_probe(A)),
    }
    write_json(report, out)
    if not report["ok"]:
        raise RuntimeError(f"Physics validation failed: {report}")
    return report


def spectral_stats(u_stack: np.ndarray) -> dict:
    norms = np.linalg.norm(u_stack.reshape(len(u_stack), -1), axis=1)
    specs = [np.fft.fftshift(np.abs(np.fft.fft2(u)) ** 2) for u in u_stack]
    mean_spec = np.mean(specs, axis=0)
    return {
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_std": float(norms.std()),
        "spectrum_energy": float(mean_spec.sum()),
    }


def load_u_array(path: Path, max_n: int = 64) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        n = min(int(f["u"].shape[0]), max_n)
        return np.asarray(f["u"][:n], dtype=np.float64)


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def cell_complete(ft_dir: Path, eval_root: Path, tag: str) -> bool:
    if not (ft_dir / "metrics.json").exists() or not (ft_dir / "checkpoint_best.pt").exists():
        return False
    for dist in ("id", "ood"):
        if not (eval_root / f"{tag}_{dist}" / "metrics.json").exists():
            return False
    return True


def scrub_incomplete_finetune(ft_dir: Path) -> bool:
    if not ft_dir.exists():
        return False
    if (ft_dir / "metrics.json").exists() and (ft_dir / "checkpoint_best.pt").exists():
        return False
    shutil.rmtree(ft_dir)
    return True


def git_sha() -> str | None:
    if not (ROOT / ".git").exists():
        return None
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def resolve_run_root(mode: str, resume: str | None, study_tag: str) -> Path:
    if resume:
        p = Path(resume)
        if not p.is_absolute():
            p = (PKG / "runs" / resume).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--resume-run not found: {p}")
        return p
    # Auto-resume incomplete Study-2 runs only (name contains study tag).
    runs = sorted((PKG / "runs").glob(f"{mode}_{study_tag}_*"), key=lambda x: x.name)
    for cand in reversed(runs):
        if not (cand / "summary.json").exists():
            return cand
    return PKG / "runs" / f"{mode}_{study_tag}_{time.strftime('%Y%m%dT%H%M%SZ')}"


def load_or_train_pretrain(
    *,
    kind: str,
    out_dir: Path,
    data_path: Path,
    pre_steps: int,
    batch: int,
    width: int,
    modes: int,
    n_layers: int,
    lr: float,
    device,
    train_cfg: dict,
    shuffle_physics: bool,
    log_status,
) -> dict:
    metrics_path = out_dir / "metrics.json"
    ckpt_path = out_dir / "checkpoint.pt"
    if metrics_path.exists() and ckpt_path.exists():
        meta = load_meta(metrics_path)
        meta["checkpoint"] = str(ckpt_path)
        log_status(f"  {kind} resume existing")
        return meta
    common = dict(
        data_path=data_path,
        out_dir=out_dir,
        steps=pre_steps,
        batch_size=batch,
        width=width,
        modes=modes,
        n_layers=n_layers,
        lr=lr,
        device=device,
        seed=0,
        shuffle_physics=shuffle_physics,
        shuffle_mode=str(train_cfg.get("shuffle_mode", "u_vs_af")),
        normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
    )
    if kind == "mml":
        return train_mml(**common)
    return train_lmop(
        **common,
        ema=float(train_cfg["ema_momentum"]),
        mask_ratio=float(train_cfg["mask_ratio"]),
        mask_blocks_min=int(train_cfg.get("mask_blocks_min", 1)),
        mask_blocks_max=int(train_cfg.get("mask_blocks_max", 3)),
        lambda_var=float(train_cfg.get("lambda_var", 25.0)),
        lambda_cov=float(train_cfg.get("lambda_cov", 1.0)),
        lambda_u=float(train_cfg.get("lambda_u", 0.1)),
        var_gamma=float(train_cfg.get("var_gamma", 1.0)),
    )


def row_from_eval(
    *,
    method: str,
    seed: int,
    n_lab: int,
    dist: str,
    ev: dict,
    ft: dict,
    pre_steps: int,
    mml_meta: dict,
    lmop_meta: dict,
    mode: str,
    tag: str,
) -> dict:
    s = ev["summary"]
    return {
        "method": method,
        "seed": seed,
        "real_label_budget": n_lab,
        "distribution": dist,
        "manufactured_family": "matched",
        "relative_l2": s["relative_l2_mean"],
        "h1_error": s["h1_error_mean"],
        "energy_error": s["energy_error_mean"],
        "flux_error": s["flux_error_mean"],
        "pde_residual": s["pde_residual_mean"],
        "parameter_count": ft["parameter_count"],
        "pretrain_steps": 0 if method == "scratch" else pre_steps,
        "pretrain_wall_seconds": 0.0
        if method == "scratch"
        else (mml_meta if method == "mml_direct" else lmop_meta)["wall_seconds"],
        "bootstrap_l2_ci_low": ev["bootstrap_relative_l2"]["ci_low"],
        "bootstrap_l2_ci_high": ev["bootstrap_relative_l2"]["ci_high"],
        "run_id": tag,
        "evidence_class": "SMOKE_NOT_EVIDENCE" if mode == "smoke" else "CONFIRMATORY_STUDY2",
        "amendment": "0003_matched_block_jepa",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "confirmatory"], default="smoke")
    parser.add_argument("--protocol", type=str, default=str(PKG / "protocol.yaml"))
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--skip-shuffled", action="store_true")
    args = parser.parse_args()

    set_threads(4)
    proto = yaml.safe_load(Path(args.protocol).read_text())
    mode = args.mode
    study_tag = f"s{proto.get('study', 2)}_{proto.get('data_version', 'v2')}"
    sizes = proto["data_sizes"][mode]
    model_cfg = proto["model"]
    train_cfg = proto["training"]

    if mode == "smoke":
        n_grid = proto["physics"]["grid_smoke"]
        width, modes, n_layers = model_cfg["smoke_width"], model_cfg["smoke_modes"], model_cfg["smoke_n_layers"]
        pre_steps = train_cfg["pretrain_steps_smoke"]
        ft_epochs = train_cfg["finetune_epochs_smoke"]
        freeze_epochs = int(train_cfg.get("freeze_epochs_smoke", 0))
        seeds = [proto["seeds"][0]]
        subsets = sizes["subsets"]
    else:
        n_grid = proto["physics"]["grid_confirmatory"]
        width, modes, n_layers = model_cfg["width"], model_cfg["modes"], model_cfg["n_layers"]
        pre_steps = train_cfg["pretrain_steps_confirmatory"]
        ft_epochs = train_cfg["finetune_epochs_confirmatory"]
        freeze_epochs = int(train_cfg.get("freeze_epochs_confirmatory", 0))
        seeds = list(proto["seeds"])
        subsets = sizes["subsets"]

    grid = DarcyGrid(n=n_grid)
    device = get_device(train_cfg["device"])
    run_root = resolve_run_root(mode, args.resume_run, study_tag)
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = PKG / "OVERNIGHT_STATUS.md"
    sha = git_sha()
    normalize_inputs = bool(train_cfg.get("normalize_inputs", True))

    def log_status(msg: str) -> None:
        with status_path.open("a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log_status(
        f"\n## Run {run_root.name}\n"
        f"- mode: {mode}\n"
        f"- study: {proto.get('study')} data_version={proto.get('data_version')}\n"
        f"- amendment: 0003_matched_block_jepa\n"
        f"- device: {device}\n"
        f"- grid: {n_grid}\n"
        f"- resume: {run_root}\n"
        f"- git_sha: {sha or 'NONE'}\n"
    )

    phys = physics_validation_report(grid, run_root / "physics_validation.json")
    log_status(f"- physics: PASSED cond≈{phys['condition_est']:.3g}")

    data_ver = str(proto.get("data_version", "v2"))
    data_dir = PKG / "data" / f"{mode}_{data_ver}"
    data_dir.mkdir(parents=True, exist_ok=True)
    skip_data = (data_dir / "manufactured_mixed.h5").exists() and (data_dir / "genuine_train.h5").exists()
    id_p = proto["permeability_id"]
    ood_p = proto["ood_primary"]
    mfg_p = proto["manufactured"]
    match_norm = bool(mfg_p.get("match_l2_norm", True))
    u_norm_range = tuple(mfg_p.get("u_norm_range", [1.8, 3.4]))

    if skip_data:
        log_status(f"- data: RESUME existing HDF5 under {data_dir.name}")
        gen_train = load_meta(data_dir / "genuine_train.meta.json")
        gen_val = load_meta(data_dir / "genuine_val.meta.json")
        gen_test = load_meta(data_dir / "genuine_test_id.meta.json")
        gen_ood = load_meta(data_dir / "genuine_ood.meta.json")
        mfg = load_meta(data_dir / "manufactured_mixed.meta.json")
        if not (data_dir / "manufactured_low.h5").exists():
            generate_manufactured_dataset(
                data_dir / "manufactured_low.h5",
                grid=grid,
                n=min(64, sizes["n_manufactured"]),
                base_seed=600_000,
                family="low",
                kmax=int(mfg_p["low_kmax"]),
                n_modes_range=tuple(mfg_p["n_modes_range"]),
                amplitude_range=tuple(mfg_p["amplitude_range"]),
                length_scale=id_p["length_scale"],
                variance=id_p["variance"],
                a_min=id_p["a_min"],
                a_max=id_p["a_max"],
                match_l2_norm=match_norm,
                u_norm_range=u_norm_range,
            )
    else:
        log_status(f"- data: GENERATE matched MMS under {data_dir.name}")
        gen_train = generate_genuine_dataset(
            data_dir / "genuine_train.h5",
            grid=grid,
            n=sizes["n_genuine_train_pool"],
            base_seed=100_000,
            length_scale=id_p["length_scale"],
            variance=id_p["variance"],
            a_min=id_p["a_min"],
            a_max=id_p["a_max"],
            family="id_train",
        )
        gen_val = generate_genuine_dataset(
            data_dir / "genuine_val.h5",
            grid=grid,
            n=sizes["n_val"],
            base_seed=200_000,
            length_scale=id_p["length_scale"],
            variance=id_p["variance"],
            a_min=id_p["a_min"],
            a_max=id_p["a_max"],
            family="id_val",
        )
        gen_test = generate_genuine_dataset(
            data_dir / "genuine_test_id.h5",
            grid=grid,
            n=sizes["n_test_id"],
            base_seed=300_000,
            length_scale=id_p["length_scale"],
            variance=id_p["variance"],
            a_min=id_p["a_min"],
            a_max=id_p["a_max"],
            family="id_test",
        )
        gen_ood = generate_genuine_dataset(
            data_dir / "genuine_ood.h5",
            grid=grid,
            n=sizes["n_ood"],
            base_seed=400_000,
            length_scale=ood_p["length_scale"],
            variance=ood_p["variance"],
            a_min=ood_p["a_min"],
            a_max=ood_p["a_max"],
            family="ood_corr",
        )
        mfg = generate_manufactured_dataset(
            data_dir / "manufactured_mixed.h5",
            grid=grid,
            n=sizes["n_manufactured"],
            base_seed=500_000,
            family=str(mfg_p.get("primary_family", "matched")),
            kmax=int(mfg_p["mixed_kmax"]),
            n_modes_range=tuple(mfg_p["n_modes_range"]),
            amplitude_range=tuple(mfg_p["amplitude_range"]),
            length_scale=id_p["length_scale"],
            variance=id_p["variance"],
            a_min=id_p["a_min"],
            a_max=id_p["a_max"],
            match_l2_norm=match_norm,
            u_norm_range=u_norm_range,
        )
        generate_manufactured_dataset(
            data_dir / "manufactured_low.h5",
            grid=grid,
            n=min(64, sizes["n_manufactured"]),
            base_seed=600_000,
            family="low",
            kmax=int(mfg_p["low_kmax"]),
            n_modes_range=tuple(mfg_p["n_modes_range"]),
            amplitude_range=tuple(mfg_p["amplitude_range"]),
            length_scale=id_p["length_scale"],
            variance=id_p["variance"],
            a_min=id_p["a_min"],
            a_max=id_p["a_max"],
            match_l2_norm=match_norm,
            u_norm_range=u_norm_range,
        )

    man_path = PKG / "manifests" / f"{mode}_{data_ver}_splits.json"
    if man_path.exists() and skip_data:
        manifest = load_meta(man_path)
        log_status(f"- manifests: RESUME {man_path.name} sha={manifest['manifest_sha256'][:12]}")
    else:
        manifest = build_split_manifest(
            genuine_train_ids=gen_train["field_ids"],
            genuine_val_ids=gen_val["field_ids"],
            genuine_test_ids=gen_test["field_ids"],
            genuine_ood_ids=gen_ood["field_ids"],
            manufactured_ids=mfg["field_ids"],
            subset_sizes=subsets,
            subset_seed=int(proto["split_seed"]),
        )
        write_json(manifest, man_path)
        log_status(f"- manifests: {man_path.name} sha={manifest['manifest_sha256'][:12]}")

    u_real = load_u_array(data_dir / "genuine_train.h5")
    u_mfg = load_u_array(data_dir / "manufactured_mixed.h5")
    u_low = load_u_array(data_dir / "manufactured_low.h5")
    mismatch = {
        "genuine": spectral_stats(u_real),
        "manufactured_matched": spectral_stats(u_mfg),
        "manufactured_low": spectral_stats(u_low),
        "norm_ratio_mfg_over_genuine": float(
            spectral_stats(u_mfg)["l2_norm_mean"] / max(1e-12, spectral_stats(u_real)["l2_norm_mean"])
        ),
        "note": "Study-2 matched MMS; ratio should be near 1.",
        "amendment": "0003_matched_block_jepa",
    }
    write_json(mismatch, PKG / "results" / "tables" / f"{mode}_v2_synthetic_real_mismatch.json")
    fig_dir = PKG / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.bar(
        ["genuine", "mfg_matched", "mfg_low"],
        [
            mismatch["genuine"]["l2_norm_mean"],
            mismatch["manufactured_matched"]["l2_norm_mean"],
            mismatch["manufactured_low"]["l2_norm_mean"],
        ],
    )
    plt.ylabel("mean ||u||_2")
    plt.title(f"{mode} v2: synthetic vs real")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{mode}_v2_synthetic_real_norms.png", dpi=140)
    plt.close()
    log_status(f"- norm_ratio mfg/genuine={mismatch['norm_ratio_mfg_over_genuine']:.3f}")

    batch = int(train_cfg["batch_size"])
    lr = float(train_cfg["lr"])
    table_rows: list[dict] = []

    log_status("- pretrain MML-direct...")
    mml_meta = load_or_train_pretrain(
        kind="mml",
        out_dir=run_root / "pretrain_mml",
        data_path=data_dir / "manufactured_mixed.h5",
        pre_steps=pre_steps,
        batch=batch,
        width=width,
        modes=modes,
        n_layers=n_layers,
        lr=lr,
        device=device,
        train_cfg=train_cfg,
        shuffle_physics=False,
        log_status=log_status,
    )
    log_status(f"  MML done loss={mml_meta.get('final_loss')} params={mml_meta.get('parameter_count')}")

    log_status("- pretrain LMOP-JEPA...")
    lmop_meta = load_or_train_pretrain(
        kind="lmop",
        out_dir=run_root / "pretrain_lmop",
        data_path=data_dir / "manufactured_mixed.h5",
        pre_steps=pre_steps,
        batch=batch,
        width=width,
        modes=modes,
        n_layers=n_layers,
        lr=lr,
        device=device,
        train_cfg=train_cfg,
        shuffle_physics=False,
        log_status=log_status,
    )
    log_status(f"  LMOP done loss={lmop_meta['final_loss']:.4f} std={lmop_meta.get('mean_latent_std')}")

    if not args.skip_shuffled:
        log_status("- shuffled-physics LMOP control (u_vs_af)...")
        shuf = load_or_train_pretrain(
            kind="lmop",
            out_dir=run_root / "pretrain_lmop_shuffled",
            data_path=data_dir / "manufactured_mixed.h5",
            pre_steps=pre_steps,
            batch=batch,
            width=width,
            modes=modes,
            n_layers=n_layers,
            lr=lr,
            device=device,
            train_cfg=train_cfg,
            shuffle_physics=True,
            log_status=log_status,
        )
        shuffle_report = {
            "shuffle_mode": train_cfg.get("shuffle_mode", "u_vs_af"),
            "correct_final_loss": lmop_meta.get("final_loss"),
            "shuffled_final_loss": shuf.get("final_loss"),
            "correct_mean_latent_std": lmop_meta.get("mean_latent_std"),
            "shuffled_mean_latent_std": shuf.get("mean_latent_std"),
            "mechanism_supported": bool(
                shuf.get("final_loss") is not None
                and lmop_meta.get("final_loss") is not None
                and float(shuf["final_loss"]) > float(lmop_meta["final_loss"])
            ),
            "correct": lmop_meta,
            "shuffled": shuf,
            "amendment": "0003_matched_block_jepa",
        }
        write_json(shuffle_report, run_root / "shuffled_control.json")
        log_status(
            f"  shuffle loss={shuf.get('final_loss')} vs correct={lmop_meta.get('final_loss')} "
            f"supported={shuffle_report['mechanism_supported']}"
        )

    methods = [
        ("scratch", None, "none"),
        ("mml_direct", Path(mml_meta["checkpoint"]), "mml"),
        ("lmop_jepa", Path(lmop_meta["checkpoint"]), "lmop"),
    ]

    for seed in seeds:
        for n_lab in subsets:
            train_idx = manifest["subsets"][f"n{n_lab}"]
            for method, ckpt, kind in methods:
                tag = f"{method}_n{n_lab}_s{seed}"
                ft_dir = run_root / "finetune" / tag
                if scrub_incomplete_finetune(ft_dir):
                    log_status(f"- scrubbed incomplete finetune {tag}")
                if cell_complete(ft_dir, run_root / "eval", tag):
                    log_status(f"- skip complete {tag}")
                    ft = load_meta(ft_dir / "metrics.json")
                    ft["checkpoint"] = str(ft_dir / "checkpoint_best.pt")
                    for dist in ("id", "ood"):
                        ev = load_meta(run_root / "eval" / f"{tag}_{dist}" / "metrics.json")
                        table_rows.append(
                            row_from_eval(
                                method=method,
                                seed=seed,
                                n_lab=n_lab,
                                dist=dist,
                                ev=ev,
                                ft=ft,
                                pre_steps=pre_steps,
                                mml_meta=mml_meta,
                                lmop_meta=lmop_meta,
                                mode=mode,
                                tag=tag,
                            )
                        )
                    continue

                log_status(f"- finetune {tag} (freeze_epochs={freeze_epochs})")
                ft = finetune_genuine(
                    data_dir / "genuine_train.h5",
                    data_dir / "genuine_val.h5",
                    ft_dir,
                    train_indices=train_idx,
                    init_checkpoint=ckpt,
                    init_kind=kind,
                    epochs=ft_epochs,
                    freeze_epochs=freeze_epochs,
                    batch_size=batch,
                    width=width,
                    modes=modes,
                    n_layers=n_layers,
                    lr=lr,
                    device=device,
                    seed=seed,
                    normalize_inputs=normalize_inputs,
                )
                for dist, path in [("id", data_dir / "genuine_test_id.h5"), ("ood", data_dir / "genuine_ood.h5")]:
                    ev_dir = run_root / "eval" / f"{tag}_{dist}"
                    if (ev_dir / "metrics.json").exists():
                        ev = load_meta(ev_dir / "metrics.json")
                    else:
                        ev = evaluate_model(
                            Path(ft["checkpoint"]),
                            path,
                            ev_dir,
                            grid=grid,
                            width=width,
                            modes=modes,
                            n_layers=n_layers,
                            device=device,
                            distribution=dist,
                            normalize_inputs=normalize_inputs,
                        )
                    table_rows.append(
                        row_from_eval(
                            method=method,
                            seed=seed,
                            n_lab=n_lab,
                            dist=dist,
                            ev=ev,
                            ft=ft,
                            pre_steps=pre_steps,
                            mml_meta=mml_meta,
                            lmop_meta=lmop_meta,
                            mode=mode,
                            tag=tag,
                        )
                    )

    if not table_rows:
        raise RuntimeError("No evaluation rows produced — matrix empty.")

    table_name = "smoke_v2_metrics.csv" if mode == "smoke" else "confirmatory_v2.csv"
    table_path = PKG / "results" / "tables" / table_name
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    plt.figure(figsize=(6, 4))
    for method in ["scratch", "mml_direct", "lmop_jepa"]:
        xs, ys = [], []
        for n_lab in subsets:
            vals = [
                r["relative_l2"]
                for r in table_rows
                if r["method"] == method and r["real_label_budget"] == n_lab and r["distribution"] == "id"
            ]
            if vals:
                xs.append(n_lab)
                ys.append(float(np.mean(vals)))
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Genuine label budget N")
    plt.ylabel("Relative L2 (ID)")
    plt.title(f"{mode} v2: label efficiency (amendment 0003)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{mode}_v2_label_efficiency.png", dpi=140)
    plt.close()

    summary = {
        "rows": table_rows,
        "run_root": str(run_root),
        "mode": mode,
        "study": proto.get("study"),
        "data_version": data_ver,
        "git_sha": sha,
        "protocol_amendments": ["0001", "0002", "0003_matched_block_jepa"],
        "norm_ratio_mfg_over_genuine": mismatch["norm_ratio_mfg_over_genuine"],
        "n_rows": len(table_rows),
    }
    write_json(summary, run_root / "summary.json")
    log_status(f"- table: {table_path}")

    if mode == "confirmatory":
        claim = write_claim_artifacts(
            table_rows=table_rows,
            run_root=run_root,
            proto=proto,
            shuffle_path=run_root / "shuffled_control.json",
            git_sha=sha,
        )
        # Also write Study-2-specific copies
        for src_name, dst_name in [
            ("claim_gate.json", "claim_gate_v2.json"),
            ("claim_gate.md", "claim_gate_v2.md"),
        ]:
            src = PKG / "results" / "tables" / src_name
            if src.exists():
                shutil.copy(src, PKG / "results" / "tables" / dst_name)
        log_status(f"- claim_gate: {claim['verdict']} -> {claim['artifact']}")
        log_status(f"- claim_detail: {claim['summary']}")

    log_status(f"- DONE {mode} study2")
    print(json.dumps({"table": str(table_path), "n_rows": len(table_rows), "run_root": str(run_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
