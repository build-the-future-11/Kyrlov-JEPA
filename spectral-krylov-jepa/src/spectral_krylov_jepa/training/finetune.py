"""Fine-tuning downstream ground-state predictors."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_krylov_jepa.data.datasets import labeled_from_manifest
from spectral_krylov_jepa.data.splits import read_manifest
from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.models.encoders import PotentialEncoder, default_encoder_kwargs
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.training.checkpointing import save_checkpoint
from spectral_krylov_jepa.training.losses import downstream_loss, wavefunction_fidelity
from spectral_krylov_jepa.training.optim import build_scheduler
from spectral_krylov_jepa.training.seed import seed_everything
from spectral_krylov_jepa.utils.device import get_device
from spectral_krylov_jepa.utils.io import environment_info, make_run_dir, write_json
from spectral_krylov_jepa.utils.logging import setup_logger


def _compute_train_stats(dataset) -> dict[str, float]:
    """Energy and potential standardization stats from the fine-tune subset only."""
    energies = []
    v_vals = []
    for i in range(len(dataset)):
        item = dataset[i]
        energies.append(float(item["energy"].item()))
        v_vals.append(item["potential"].numpy().ravel())
    e = np.asarray(energies, dtype=np.float64)
    v = np.concatenate(v_vals).astype(np.float64)
    return {
        "energy_mean": float(e.mean()),
        "energy_std": float(max(e.std(), 1e-6)),
        "v_center": float(v.mean()),
        "v_scale": float(max(v.std(), 1e-6)),
    }


@torch.no_grad()
def evaluate_loader(
    model: DownstreamGroundStateModel,
    loader: DataLoader,
    device: torch.device,
    cell_area_val: float,
    *,
    lambda_psi: float = 1.0,
    lambda_e: float = 1.0,
    lambda_psi_mse: float = 0.1,
) -> dict[str, float]:
    model.eval()
    fidelities = []
    energy_errs = []
    losses = []
    for batch in loader:
        v = batch["potential"].to(device)
        psi = batch["psi0"].to(device)
        e = batch["energy"].to(device)
        out = model(v)
        loss, _ = downstream_loss(
            out.energy,
            e,
            out.psi,
            psi,
            cell_area_val,
            lambda_psi=lambda_psi,
            lambda_e=lambda_e,
            lambda_psi_mse=lambda_psi_mse,
            energy_mean=model.energy_head.energy_mean if model.energy_head.use_standardization else None,
            energy_std=model.energy_head.energy_std if model.energy_head.use_standardization else None,
        )
        f = wavefunction_fidelity(out.psi, psi, cell_area_val)
        rel = (out.energy - e).abs() / (e.abs() + 1e-6)
        fidelities.extend(f.cpu().tolist())
        energy_errs.extend(rel.cpu().tolist())
        losses.append(float(loss.item()))
    fid_mean = float(np.mean(fidelities)) if fidelities else float("nan")
    e_err = float(np.mean(energy_errs)) if energy_errs else float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "fidelity_mean": fid_mean,
        "fidelity_std": float(np.std(fidelities)) if fidelities else float("nan"),
        "rel_energy_error_mean": e_err,
        # Selection score: prefer high fidelity and low energy error (a priori weights)
        "selection_score": (1.0 - fid_mean) + 0.5 * e_err,
    }


def finetune(
    *,
    data_path: str | Path,
    manifest_path: str | Path,
    subset: str = "n10",
    encoder_path: str | Path | None = None,
    method_name: str = "scratch",
    run_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune (or train from scratch) a downstream ground-state model."""
    cfg = dict(config or {})
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))

    root = Path(cfg.get("experiments_root", "experiments/raw"))
    run_path = Path(run_dir) if run_dir else make_run_dir(root, f"finetune_{method_name}_{subset}")
    run_path.mkdir(parents=True, exist_ok=True)
    log = setup_logger("finetune", log_file=run_path / "train.log")

    manifest = read_manifest(manifest_path)
    train_ds = labeled_from_manifest(data_path, manifest, subset)
    val_ds = labeled_from_manifest(data_path, manifest, "validation")

    img_size = int(cfg.get("img_size", train_ds.ny))
    grid = GridSpec(n_interior=img_size)
    ca = cell_area(grid)

    size = cfg.get("model_size", "default")
    model = DownstreamGroundStateModel(
        img_size=img_size,
        size=size,
        cell_area=ca,
    ).to(device)

    stats = _compute_train_stats(train_ds)
    if bool(cfg.get("standardize_energy", True)):
        model.set_energy_stats(stats["energy_mean"], stats["energy_std"])
    if bool(cfg.get("standardize_potential", True)):
        model.set_potential_stats(stats["v_center"], stats["v_scale"])
    log.info("Train stats: %s", stats)

    if encoder_path is not None:
        payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
        enc_state = payload["encoder"] if "encoder" in payload else payload
        kwargs = default_encoder_kwargs(size)
        enc = PotentialEncoder(img_size=img_size, **kwargs)
        enc.load_state_dict(enc_state, strict=True)
        model.load_pretrained_encoder(enc)
        log.info("Loaded pretrained encoder from %s", encoder_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(cfg.get("batch_size", 4)), len(train_ds)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(int(cfg.get("batch_size", 4)), max(1, len(val_ds))),
        shuffle=False,
        num_workers=0,
    )

    # Differential LRs: slightly lower for pretrained encoder
    lr = float(cfg.get("lr", 1e-3))
    enc_lr = float(cfg.get("encoder_lr", lr * 0.3 if encoder_path else lr))
    from torch.optim import AdamW

    opt = AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": enc_lr},
            {
                "params": list(model.energy_head.parameters()) + list(model.psi_decoder.parameters()),
                "lr": lr,
            },
        ],
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    epochs = int(cfg.get("epochs", 50))
    steps_per_epoch = max(1, len(train_loader))
    sched = build_scheduler(
        opt,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=max(1, steps_per_epoch),
    )

    write_json(
        {
            "config": cfg,
            "method": method_name,
            "subset": subset,
            "encoder_path": str(encoder_path) if encoder_path else None,
            "env": environment_info(str(device)),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "train_stats": stats,
        },
        run_path / "config_snapshot.json",
    )

    lambda_psi = float(cfg.get("lambda_psi", 1.0))
    lambda_e = float(cfg.get("lambda_e", 1.0))
    lambda_psi_mse = float(cfg.get("lambda_psi_mse", 0.1))
    patience = int(cfg.get("early_stopping_patience", 15))
    best_score = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    t0 = time.time()

    e_mean = model.energy_head.energy_mean
    e_std = model.energy_head.energy_std
    use_z = model.energy_head.use_standardization

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            v = batch["potential"].to(device)
            psi = batch["psi0"].to(device)
            e = batch["energy"].to(device)
            out = model(v)
            loss, stats_b = downstream_loss(
                out.energy,
                e,
                out.psi,
                psi,
                ca,
                lambda_psi=lambda_psi,
                lambda_e=lambda_e,
                lambda_psi_mse=lambda_psi_mse,
                energy_mean=e_mean if use_z else None,
                energy_std=e_std if use_z else None,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            if sched is not None:
                sched.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            ca,
            lambda_psi=lambda_psi,
            lambda_e=lambda_e,
            lambda_psi_mse=lambda_psi_mse,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        log.info(
            "epoch=%d train_loss=%.4f val_fid=%.4f val_Eerr=%.4f score=%.4f",
            epoch,
            row["train_loss"],
            val_metrics["fidelity_mean"],
            val_metrics["rel_energy_error_mean"],
            val_metrics["selection_score"],
        )

        if val_metrics["selection_score"] < best_score - 1e-6:
            best_score = val_metrics["selection_score"]
            best_epoch = epoch
            save_checkpoint(
                run_path / "checkpoint_best.pt",
                model,
                optimizer=opt,
                epoch=epoch,
                metrics=val_metrics,
                config={**cfg, "train_stats": stats},
            )

        if epoch - best_epoch >= patience:
            log.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
            break

    best_path = run_path / "checkpoint_best.pt"
    if best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])

    final_val = evaluate_loader(
        model,
        val_loader,
        device,
        ca,
        lambda_psi=lambda_psi,
        lambda_e=lambda_e,
        lambda_psi_mse=lambda_psi_mse,
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "final_val": final_val,
        "train_stats": stats,
        "elapsed_sec": time.time() - t0,
        "history": history,
    }
    write_json(metrics, run_path / "metrics.json")
    write_json({"status": "ok", "failure": None}, run_path / "status.json")
    save_checkpoint(run_path / "checkpoint_last.pt", model, epoch=epochs, metrics=final_val, config=cfg)
    train_ds.close()
    val_ds.close()
    log.info("Finetune complete: %s", final_val)
    return {"run_dir": str(run_path), "metrics": metrics, "checkpoint": str(best_path)}
