"""Pretraining loops for Field / Operator / Krylov JEPA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral_krylov_jepa.data.datasets import UnlabeledKrylovDataset
from spectral_krylov_jepa.models.field_jepa import FieldJEPA
from spectral_krylov_jepa.models.krylov_jepa import KrylovJEPA
from spectral_krylov_jepa.models.operator_jepa import OperatorJEPA
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.training.checkpointing import save_checkpoint
from spectral_krylov_jepa.training.optim import build_optimizer, build_scheduler
from spectral_krylov_jepa.training.seed import seed_everything
from spectral_krylov_jepa.utils.device import get_device
from spectral_krylov_jepa.utils.io import environment_info, make_run_dir, write_json
from spectral_krylov_jepa.utils.logging import setup_logger

MethodName = Literal["field", "operator", "krylov"]


def _hq_normalized_batch(
    potentials: torch.Tensor,
    q: torch.Tensor,
    grid: GridSpec,
) -> torch.Tensor:
    """Compute normalize(Hq) on CPU with sparse H, return torch tensor on same device as q."""
    device = q.device
    dtype = q.dtype
    B = q.shape[0]
    n_dof = grid.n_dof
    out = torch.zeros(B, n_dof, device=device, dtype=dtype)
    v_np = potentials.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    for i in range(B):
        ham = build_hamiltonian(grid, v_np[i])
        hq = ham.matvec(q_np[i].reshape(-1))
        nrm = np.linalg.norm(hq)
        if nrm < 1e-12:
            hq = q_np[i].reshape(-1)
        else:
            hq = hq / nrm
        out[i] = torch.from_numpy(hq.astype(np.float32))
    return out.to(device=device, dtype=dtype)


def build_pretrain_model(
    method: MethodName,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    img_size = int(cfg.get("img_size", 32))
    size = cfg.get("model_size", "default")
    ema = float(cfg.get("ema_momentum", 0.996))
    overrides = dict(cfg.get("encoder_overrides", {}))
    if method == "field":
        return FieldJEPA(
            img_size=img_size,
            size=size,
            mask_ratio=float(cfg.get("mask_ratio", 0.4)),
            ema_momentum=ema,
            **overrides,
        )
    if method == "operator":
        return OperatorJEPA(img_size=img_size, size=size, ema_momentum=ema, **overrides)
    if method == "krylov":
        return KrylovJEPA(
            img_size=img_size,
            size=size,
            context_steps=int(cfg.get("context_steps", 2)),
            ema_momentum=ema,
            lambda_coeff=float(cfg.get("lambda_coeff", 0.0)),
            lambda_projected_ritz=float(cfg.get("lambda_projected_ritz", 0.0)),
            projected_modes=int(cfg.get("projected_modes", 9)),
            remove_v=bool(cfg.get("remove_v", False)),
            normalize_latents=bool(cfg.get("normalize_latents", True)),
            **overrides,
        )
    raise ValueError(f"Unknown method {method}")


def pretrain(
    *,
    method: MethodName,
    data_path: str | Path,
    run_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a JEPA pretraining job and write artifacts under ``run_dir``."""
    cfg = dict(config or {})
    if int(cfg.get("target_update_frequency", 1)) < 1:
        raise ValueError("target_update_frequency must be positive")
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))
    logger = setup_logger("pretrain")

    root = Path(cfg.get("experiments_root", "experiments/raw"))
    run_path = Path(run_dir) if run_dir else make_run_dir(root, f"pretrain_{method}")
    run_path.mkdir(parents=True, exist_ok=True)
    log = setup_logger("pretrain", log_file=run_path / "train.log")

    dataset = UnlabeledKrylovDataset(
        data_path,
        remove_v=bool(cfg.get("remove_v", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=True,
    )
    if len(loader) == 0 or int(cfg.get("max_steps", 500)) < 1:
        dataset.close()
        raise ValueError("Pretraining needs a full batch and at least one step")
    grid = GridSpec(n_interior=int(cfg.get("img_size", dataset.ny)))
    model = build_pretrain_model(method, cfg).to(device)
    # Exclude EMA target params from optimizer (already requires_grad=False)
    opt = build_optimizer(
        model,
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    max_steps = int(cfg.get("max_steps", 500))
    sched = build_scheduler(
        opt,
        total_steps=max_steps,
        warmup_steps=int(cfg.get("warmup_steps", max(1, max_steps // 10))),
    )

    write_json(
        {
            "config": cfg,
            "method": method,
            "env": environment_info(str(device)),
            "data_path": str(data_path),
            "n_params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        run_path / "config_snapshot.json",
    )

    model.train()
    step = 0
    losses: list[float] = []
    t0 = time.time()
    context_steps = int(cfg.get("context_steps", 2))
    pbar = tqdm(total=max_steps, desc=f"pretrain:{method}")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            v = batch["potential"].to(device)
            q = batch["q"].to(device)  # (B, depth+1, n_dof)
            alpha = batch["alpha"].to(device)
            beta = batch["beta"].to(device)

            if method == "field":
                out = model(v)
                loss = out.loss
            elif method == "operator":
                q0 = q[:, 0]
                hq = _hq_normalized_batch(v, q0, grid)
                out = model(v, q0, hq)
                loss = out.loss
            else:
                # krylov
                depth_avail = q.shape[1] - 1
                if context_steps > depth_avail:
                    raise ValueError(
                        f"context_steps={context_steps} requires depth>={context_steps}, got {depth_avail}"
                    )
                q_ctx = q[:, :context_steps]
                q_tgt = q[:, context_steps]
                out = model(v, q_ctx, q_tgt, alpha=alpha, beta=beta)
                loss = out.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            opt.step()
            if sched is not None:
                sched.step()
            if (step + 1) % int(cfg.get("target_update_frequency", 1)) == 0:
                model.update_target()

            losses.append(float(loss.item()))
            step += 1
            pbar.update(1)
            if step % int(cfg.get("log_every", 20)) == 0:
                log.info("step=%d loss=%.6f", step, losses[-1])

    pbar.close()
    elapsed = time.time() - t0
    metrics = {
        "final_loss": losses[-1] if losses else float("nan"),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": step,
        "elapsed_sec": elapsed,
        "loss_history": losses,
        "n_params_total": sum(p.numel() for p in model.parameters()),
        "n_params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder_parameters": sum(p.numel() for p in model.potential_encoder().parameters()),
    }
    save_checkpoint(
        run_path / "checkpoint_last.pt",
        model,
        optimizer=opt,
        scheduler=sched,
        step=step,
        metrics=metrics,
        config=cfg,
    )
    # Also save potential encoder only for downstream transfer
    enc = model.potential_encoder()
    torch.save({"encoder": enc.state_dict(), "method": method, "config": cfg}, run_path / "encoder.pt")
    write_json(metrics, run_path / "metrics.json")
    write_json({"status": "ok", "failure": None}, run_path / "status.json")
    log.info("Pretrain complete: %s", metrics)
    dataset.close()
    return {"run_dir": str(run_path), "metrics": metrics, "encoder_path": str(run_path / "encoder.pt")}
