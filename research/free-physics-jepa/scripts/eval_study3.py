#!/usr/bin/env python3
"""Study-3 breakthrough eval (amendment 0004).

Reuses Study-2 matched data + pretrained checkpoints, evaluates under
zero_shot / probe / low_lr_ft adaptation protocols.
Optionally retrains LMOP with rebalanced λ_var=1, λ_u=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "lmop_jepa"))
sys.path.insert(0, str(PKG / "scripts"))

from darcy_reference import DarcyGrid  # noqa: E402
from lmop_jepa.data import write_json  # noqa: E402
from lmop_jepa.models import DownstreamSolver, LMOPJEPA  # noqa: E402
from lmop_jepa.training import (  # noqa: E402
    evaluate_model,
    finetune_genuine,
    get_device,
    normalize_af,
    set_threads,
    train_lmop,
)
from aggregate_claim_gate import write_claim_artifacts  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PKG.parents[1], text=True).strip()
    except Exception:
        return None


def build_downstream_from_pretrain(
    *,
    kind: str,
    ckpt: Path | None,
    width: int,
    modes: int,
    n_layers: int,
    device,
) -> DownstreamSolver:
    model = DownstreamSolver(width=width, modes=modes, n_layers=n_layers).to(device)
    if kind in ("none", "scratch") or ckpt is None:
        return model
    payload = __import__("torch").load(ckpt, map_location="cpu", weights_only=False)
    if kind == "mml":
        model.load_state_dict(payload["model"])
    elif kind == "lmop":
        lmop = LMOPJEPA(width=width, modes=modes, n_layers=n_layers)
        lmop.load_state_dict(payload["model"], strict=True)
        lmop.transfer_to_downstream(model)
    else:
        raise ValueError(kind)
    return model


def zero_shot_eval(
    *,
    model: DownstreamSolver,
    data_path: Path,
    out_dir: Path,
    grid: DarcyGrid,
    width: int,
    modes: int,
    n_layers: int,
    device,
    distribution: str,
    normalize_inputs: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint_best.pt"
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "config": {"width": width, "modes": modes, "n_layers": n_layers},
            "normalize_inputs": normalize_inputs,
            "adaptation": "zero_shot",
        },
        ckpt,
    )
    return evaluate_model(
        ckpt,
        data_path,
        out_dir,
        grid=grid,
        width=width,
        modes=modes,
        n_layers=n_layers,
        device=device,
        distribution=distribution,
        normalize_inputs=normalize_inputs,
    )


def row_from(
    *,
    method: str,
    seed: int,
    n_lab: int,
    dist: str,
    ev: dict,
    adaptation: str,
    pre_steps: int,
    tag: str,
) -> dict:
    s = ev["summary"]
    return {
        "method": method,
        "seed": seed,
        "real_label_budget": n_lab,
        "distribution": dist,
        "adaptation": adaptation,
        "manufactured_family": "matched",
        "relative_l2": s["relative_l2_mean"],
        "h1_error": s["h1_error_mean"],
        "energy_error": s["energy_error_mean"],
        "flux_error": s["flux_error_mean"],
        "pde_residual": s["pde_residual_mean"],
        "pretrain_steps": pre_steps if method != "scratch" else 0,
        "bootstrap_l2_ci_low": ev["bootstrap_relative_l2"]["ci_low"],
        "bootstrap_l2_ci_high": ev["bootstrap_relative_l2"]["ci_high"],
        "run_id": tag,
        "evidence_class": "CONFIRMATORY_STUDY3",
        "amendment": "0004_preserve_ssl",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "confirmatory"], default="confirmatory")
    parser.add_argument(
        "--pretrain-run",
        type=str,
        default="confirmatory_s2_v2_20260926T162730Z",
        help="Run dir with Study-2 (or Study-3) pretrain_mml / pretrain_lmop checkpoints",
    )
    parser.add_argument(
        "--adaptations",
        type=str,
        default="zero_shot,probe,low_lr_ft",
        help="Comma list: zero_shot,probe,low_lr_ft",
    )
    parser.add_argument(
        "--retrain-lmop",
        action="store_true",
        help="Retrain LMOP with Study-3 λ_var=1, λ_u=1 before eval",
    )
    parser.add_argument("--skip-shuffled-retrain", action="store_true")
    args = parser.parse_args()

    set_threads(4)
    proto = yaml.safe_load((PKG / "protocol.yaml").read_text())
    mode = args.mode
    sizes = proto["data_sizes"][mode]
    model_cfg = proto["model"]
    train_cfg = proto["training"]
    data_ver = str(proto.get("data_version", "v2"))
    data_dir = PKG / "data" / f"{mode}_{data_ver}"
    man_path = PKG / "manifests" / f"{mode}_{data_ver}_splits.json"
    if not man_path.exists():
        raise FileNotFoundError(man_path)
    manifest = load_json(man_path)

    if mode == "smoke":
        n_grid = proto["physics"]["grid_smoke"]
        width, modes, n_layers = model_cfg["smoke_width"], model_cfg["smoke_modes"], model_cfg["smoke_n_layers"]
        pre_steps = train_cfg["pretrain_steps_smoke"]
        ft_epochs = train_cfg["finetune_epochs_smoke"]
        seeds = [proto["seeds"][0]]
        subsets = sizes["subsets"]
    else:
        n_grid = proto["physics"]["grid_confirmatory"]
        width, modes, n_layers = model_cfg["width"], model_cfg["modes"], model_cfg["n_layers"]
        pre_steps = train_cfg["pretrain_steps_confirmatory"]
        ft_epochs = train_cfg["finetune_epochs_confirmatory"]
        seeds = list(proto["seeds"])
        subsets = sizes["subsets"]

    grid = DarcyGrid(n=n_grid)
    device = get_device(train_cfg["device"])
    normalize_inputs = bool(train_cfg.get("normalize_inputs", True))
    backbone_lr_scale = float(train_cfg.get("backbone_lr_scale", 0.05))
    adaptations = [a.strip() for a in args.adaptations.split(",") if a.strip()]

    pre_root = PKG / "runs" / args.pretrain_run
    run_root = PKG / "runs" / f"{mode}_s3_v2_{time.strftime('%Y%m%dT%H%M%SZ')}"
    run_root.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    status = PKG / "OVERNIGHT_STATUS.md"

    def log(msg: str) -> None:
        with status.open("a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log(
        f"\n## Study-3 run {run_root.name}\n"
        f"- amendment: 0004_preserve_ssl\n"
        f"- pretrain_source: {pre_root.name}\n"
        f"- adaptations: {adaptations}\n"
        f"- device: {device}\n"
        f"- git_sha: {sha}\n"
    )

    mml_ckpt = pre_root / "pretrain_mml" / "checkpoint.pt"
    lmop_dir = run_root / "pretrain_lmop"
    if args.retrain_lmop:
        log("- retrain LMOP with λ_var=1, λ_u=1...")
        lmop_meta = train_lmop(
            data_dir / "manufactured_mixed.h5",
            lmop_dir,
            steps=pre_steps,
            batch_size=int(train_cfg["batch_size"]),
            width=width,
            modes=modes,
            n_layers=n_layers,
            lr=float(train_cfg["lr"]),
            device=device,
            seed=0,
            ema=float(train_cfg["ema_momentum"]),
            mask_ratio=float(train_cfg["mask_ratio"]),
            mask_blocks_min=int(train_cfg.get("mask_blocks_min", 1)),
            mask_blocks_max=int(train_cfg.get("mask_blocks_max", 3)),
            lambda_var=float(train_cfg["lambda_var"]),
            lambda_cov=float(train_cfg["lambda_cov"]),
            lambda_u=float(train_cfg["lambda_u"]),
            var_gamma=float(train_cfg.get("var_gamma", 1.0)),
            normalize_inputs=normalize_inputs,
        )
        log(f"  LMOP retrain done loss={lmop_meta['final_loss']:.4f} std={lmop_meta.get('mean_latent_std')}")
        if not args.skip_shuffled_retrain:
            log("- retrain shuffled LMOP (u_vs_af)...")
            shuf = train_lmop(
                data_dir / "manufactured_mixed.h5",
                run_root / "pretrain_lmop_shuffled",
                steps=pre_steps,
                batch_size=int(train_cfg["batch_size"]),
                width=width,
                modes=modes,
                n_layers=n_layers,
                lr=float(train_cfg["lr"]),
                device=device,
                seed=0,
                ema=float(train_cfg["ema_momentum"]),
                mask_ratio=float(train_cfg["mask_ratio"]),
                shuffle_physics=True,
                shuffle_mode=str(train_cfg.get("shuffle_mode", "u_vs_af")),
                mask_blocks_min=int(train_cfg.get("mask_blocks_min", 1)),
                mask_blocks_max=int(train_cfg.get("mask_blocks_max", 3)),
                lambda_var=float(train_cfg["lambda_var"]),
                lambda_cov=float(train_cfg["lambda_cov"]),
                lambda_u=float(train_cfg["lambda_u"]),
                var_gamma=float(train_cfg.get("var_gamma", 1.0)),
                normalize_inputs=normalize_inputs,
            )
            shuffle_report = {
                "shuffle_mode": train_cfg.get("shuffle_mode", "u_vs_af"),
                "correct_final_loss": lmop_meta.get("final_loss"),
                "shuffled_final_loss": shuf.get("final_loss"),
                "mechanism_supported": float(shuf["final_loss"]) > float(lmop_meta["final_loss"]),
                "correct": lmop_meta,
                "shuffled": shuf,
                "amendment": "0004_preserve_ssl",
            }
            write_json(shuffle_report, run_root / "shuffled_control.json")
            log(
                f"  shuffle {shuf['final_loss']:.4f} vs correct {lmop_meta['final_loss']:.4f} "
                f"supported={shuffle_report['mechanism_supported']}"
            )
        lmop_ckpt = Path(lmop_meta["checkpoint"])
    else:
        lmop_ckpt = pre_root / "pretrain_lmop" / "checkpoint.pt"
        # Reuse Study-2 shuffle report if present
        src_shuf = pre_root / "shuffled_control.json"
        if src_shuf.exists():
            import shutil

            shutil.copy(src_shuf, run_root / "shuffled_control.json")
        log(f"- reuse LMOP ckpt {lmop_ckpt}")

    if not mml_ckpt.exists():
        raise FileNotFoundError(mml_ckpt)
    if not lmop_ckpt.exists():
        raise FileNotFoundError(lmop_ckpt)

    methods = [
        ("scratch", None, "scratch"),
        ("mml_direct", mml_ckpt, "mml"),
        ("lmop_jepa", lmop_ckpt, "lmop"),
    ]

    all_rows: list[dict] = []
    for adaptation in adaptations:
        log(f"\n### adaptation={adaptation}")
        table_rows: list[dict] = []
        for seed in seeds:
            for n_lab in subsets:
                train_idx = manifest["subsets"][f"n{n_lab}"]
                for method, ckpt, kind in methods:
                    tag = f"{adaptation}_{method}_n{n_lab}_s{seed}"
                    if adaptation == "zero_shot":
                        # Scratch zero-shot = random; pretrained methods use pretrain head
                        model = build_downstream_from_pretrain(
                            kind=kind, ckpt=ckpt, width=width, modes=modes, n_layers=n_layers, device=device
                        )
                        for dist, path in [
                            ("id", data_dir / "genuine_test_id.h5"),
                            ("ood", data_dir / "genuine_ood.h5"),
                        ]:
                            ev = zero_shot_eval(
                                model=model,
                                data_path=path,
                                out_dir=run_root / "eval" / f"{tag}_{dist}",
                                grid=grid,
                                width=width,
                                modes=modes,
                                n_layers=n_layers,
                                device=device,
                                distribution=dist,
                                normalize_inputs=normalize_inputs,
                            )
                            table_rows.append(
                                row_from(
                                    method=method,
                                    seed=seed,
                                    n_lab=n_lab,
                                    dist=dist,
                                    ev=ev,
                                    adaptation=adaptation,
                                    pre_steps=pre_steps,
                                    tag=tag,
                                )
                            )
                        log(f"- zero_shot {tag} id_l2={table_rows[-2]['relative_l2']:.4f}")
                        continue

                    ft_dir = run_root / "finetune" / tag
                    if (ft_dir / "metrics.json").exists() and (ft_dir / "checkpoint_best.pt").exists():
                        log(f"- skip complete {tag}")
                        ft = load_json(ft_dir / "metrics.json")
                    else:
                        log(f"- finetune {tag}")
                        ft = finetune_genuine(
                            data_dir / "genuine_train.h5",
                            data_dir / "genuine_val.h5",
                            ft_dir,
                            train_indices=train_idx,
                            init_checkpoint=ckpt,
                            init_kind=kind,
                            epochs=ft_epochs,
                            freeze_epochs=int(train_cfg.get("freeze_epochs_confirmatory", 10)),
                            batch_size=int(train_cfg["batch_size"]),
                            width=width,
                            modes=modes,
                            n_layers=n_layers,
                            lr=float(train_cfg["lr"]),
                            device=device,
                            seed=seed,
                            normalize_inputs=normalize_inputs,
                            adaptation=adaptation,
                            backbone_lr_scale=backbone_lr_scale,
                        )
                    for dist, path in [
                        ("id", data_dir / "genuine_test_id.h5"),
                        ("ood", data_dir / "genuine_ood.h5"),
                    ]:
                        ev_dir = run_root / "eval" / f"{tag}_{dist}"
                        if (ev_dir / "metrics.json").exists():
                            ev = load_json(ev_dir / "metrics.json")
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
                            row_from(
                                method=method,
                                seed=seed,
                                n_lab=n_lab,
                                dist=dist,
                                ev=ev,
                                adaptation=adaptation,
                                pre_steps=pre_steps,
                                tag=tag,
                            )
                        )

        # Per-adaptation table + claim gate
        csv_path = PKG / "results" / "tables" / f"study3_{mode}_{adaptation}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            w.writeheader()
            w.writerows(table_rows)

        # Claim gate expects confirmatory seeds/subsets; filter to mml/lmop/scratch rows
        (run_root / adaptation).mkdir(parents=True, exist_ok=True)
        claim = write_claim_artifacts(
            table_rows=table_rows,
            run_root=run_root / adaptation,
            proto=proto,
            shuffle_path=run_root / "shuffled_control.json",
            git_sha=sha,
        )
        # Rename generic claim outputs to adaptation-specific
        import shutil

        for src, dst in [
            (PKG / "results" / "tables" / "claim_gate.json", PKG / "results" / "tables" / f"claim_gate_s3_{adaptation}.json"),
            (PKG / "results" / "tables" / "claim_gate.md", PKG / "results" / "tables" / f"claim_gate_s3_{adaptation}.md"),
        ]:
            if src.exists():
                shutil.copy(src, dst)
        write_json({"rows": table_rows, "verdict": claim["verdict"]}, run_root / adaptation / "summary.json")
        log(f"- {adaptation} claim_gate: {claim['verdict']} ({claim['summary']})")
        all_rows.extend(table_rows)

    # Combined summary
    combined = PKG / "results" / "tables" / f"study3_{mode}_all.csv"
    with combined.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Human summary markdown
    lines = [
        "# Study-3 results (amendment 0004)",
        "",
        f"Run: `{run_root.name}`",
        f"Pretrain source: `{pre_root.name}`",
        f"Retrain LMOP: `{args.retrain_lmop}`",
        f"Git: `{sha}`",
        "",
    ]
    for adaptation in adaptations:
        sub = [r for r in all_rows if r["adaptation"] == adaptation]
        lines.append(f"## {adaptation}")
        lines.append("")
        lines.append("| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) |")
        lines.append("|---|------|---------|-----|------|-------------|")
        for n in subsets:
            for dist in ("id", "ood"):
                means = {}
                for method in ("scratch", "mml_direct", "lmop_jepa"):
                    vals = [
                        float(r["relative_l2"])
                        for r in sub
                        if r["method"] == method and int(r["real_label_budget"]) == n and r["distribution"] == dist
                    ]
                    means[method] = float(np.mean(vals)) if vals else float("nan")
                delta = means["lmop_jepa"] - means["mml_direct"]
                lines.append(
                    f"| {n} | {dist} | {means['scratch']:.4f} | {means['mml_direct']:.4f} | "
                    f"{means['lmop_jepa']:.4f} | {delta:+.4f} |"
                )
        gate = PKG / "results" / "tables" / f"claim_gate_s3_{adaptation}.md"
        if gate.exists():
            verdict_line = next((ln for ln in gate.read_text().splitlines() if "Verdict" in ln), "")
            lines.append("")
            lines.append(verdict_line)
        lines.append("")
    summary_md = PKG / "results" / "tables" / "STUDY3_RESULTS.md"
    summary_md.write_text("\n".join(lines) + "\n")
    (PKG / "STUDY3_RESULTS.md").write_text("\n".join(lines) + "\n")
    (PKG / "paper" / "RESULTS_STUDY3.md").write_text("\n".join(lines) + "\n")
    log(f"- DONE study3 -> {summary_md}")
    print(json.dumps({"run_root": str(run_root), "n_rows": len(all_rows), "summary": str(summary_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
