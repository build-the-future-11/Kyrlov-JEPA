"""Training loops for MML-direct, LMOP-JEPA v2, and genuine fine-tuning."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lmop_jepa.evaluation import evaluate_example, paired_bootstrap_ci
from lmop_jepa.models import DownstreamSolver, LMOPJEPA, count_params
from lmop_jepa.physics import DarcyGrid


def set_threads(n: int = 4) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_af(a: torch.Tensor, f: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-instance input normalization (amendment 0003)."""
    a_n = a / (a.mean(dim=(-2, -1), keepdim=True) + eps)
    f_rms = torch.sqrt(torch.mean(f * f, dim=(-2, -1), keepdim=True) + eps)
    f_n = f / f_rms
    return a_n, f_n


class H5FieldDataset(Dataset):
    def __init__(
        self,
        path: Path,
        indices: list[int] | None = None,
        shuffle_physics: bool = False,
        shuffle_seed: int = 0,
        shuffle_mode: str = "u_vs_af",
    ):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.n = int(f.attrs.get("n", f["a"].shape[0]))
        self.indices = list(range(self.n)) if indices is None else list(indices)
        self.shuffle_physics = shuffle_physics
        self.shuffle_mode = shuffle_mode
        self.perm = np.arange(self.n)
        if shuffle_physics:
            rng = np.random.default_rng(shuffle_seed)
            self.perm = rng.permutation(self.n)
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.indices[i]
        f = self._file()
        a = np.asarray(f["a"][idx], dtype=np.float32)
        if self.shuffle_physics:
            j = int(self.perm[idx])
            if self.shuffle_mode == "u_vs_af":
                # Keep (a,f) paired; replace u only — breaks PDE consistency.
                ff = np.asarray(f["f"][idx], dtype=np.float32)
                u = np.asarray(f["u"][j], dtype=np.float32)
            elif self.shuffle_mode == "af_vs_u_legacy":
                # Study-1 broken control (kept for reference only).
                ff = np.asarray(f["f"][j], dtype=np.float32)
                u = np.asarray(f["u"][j], dtype=np.float32)
            else:
                raise ValueError(f"Unknown shuffle_mode={self.shuffle_mode}")
        else:
            ff = np.asarray(f["f"][idx], dtype=np.float32)
            u = np.asarray(f["u"][idx], dtype=np.float32)
        return {
            "a": torch.from_numpy(a),
            "f": torch.from_numpy(ff),
            "u": torch.from_numpy(u),
            "index": idx,
        }

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    num = torch.sum(diff * diff, dim=1)
    den = torch.sum(target.reshape(target.shape[0], -1) ** 2, dim=1) + 1e-8
    return torch.mean(num / den)


def train_mml(
    data_path: Path,
    out_dir: Path,
    *,
    steps: int,
    batch_size: int,
    width: int,
    modes: int,
    n_layers: int,
    lr: float,
    device: torch.device,
    seed: int,
    shuffle_physics: bool = False,
    shuffle_mode: str = "u_vs_af",
    normalize_inputs: bool = True,
) -> dict[str, Any]:
    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = H5FieldDataset(
        data_path, shuffle_physics=shuffle_physics, shuffle_seed=seed, shuffle_mode=shuffle_mode
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = DownstreamSolver(width=width, modes=modes, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    step = 0
    losses = []
    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            a, f, u = batch["a"].to(device), batch["f"].to(device), batch["u"].to(device)
            if normalize_inputs:
                a, f = normalize_af(a, f)
            pred = model(a, f)
            loss = relative_mse(pred, u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
    wall = time.time() - t0
    ckpt = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"width": width, "modes": modes, "n_layers": n_layers},
            "normalize_inputs": normalize_inputs,
            "amendment": "0003_matched_block_jepa",
        },
        ckpt,
    )
    meta = {
        "method": "mml_direct",
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "wall_seconds": wall,
        "parameter_count": count_params(model),
        "shuffle_physics": shuffle_physics,
        "shuffle_mode": shuffle_mode if shuffle_physics else None,
        "normalize_inputs": normalize_inputs,
        "checkpoint": str(ckpt),
        "amendment": "0003_matched_block_jepa",
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    ds.close()
    return meta


def train_lmop(
    data_path: Path,
    out_dir: Path,
    *,
    steps: int,
    batch_size: int,
    width: int,
    modes: int,
    n_layers: int,
    lr: float,
    device: torch.device,
    seed: int,
    ema: float,
    mask_ratio: float,
    shuffle_physics: bool = False,
    shuffle_mode: str = "u_vs_af",
    normalize_inputs: bool = True,
    collapse_std_min: float = 1e-4,
    mask_blocks_min: int = 1,
    mask_blocks_max: int = 3,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    lambda_u: float = 0.1,
    var_gamma: float = 1.0,
) -> dict[str, Any]:
    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = H5FieldDataset(
        data_path, shuffle_physics=shuffle_physics, shuffle_seed=seed, shuffle_mode=shuffle_mode
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = LMOPJEPA(
        width=width,
        modes=modes,
        n_layers=n_layers,
        ema_momentum=ema,
        mask_ratio=mask_ratio,
        mask_blocks_min=mask_blocks_min,
        mask_blocks_max=mask_blocks_max,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
        lambda_u=lambda_u,
        var_gamma=var_gamma,
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    t0 = time.time()
    step = 0
    losses = []
    stds = []
    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            a, f, u = batch["a"].to(device), batch["f"].to(device), batch["u"].to(device)
            if normalize_inputs:
                a, f = normalize_af(a, f)
            loss, diag = model(a, f, u)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite LMOP loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target()
            losses.append(float(loss.item()))
            stds.append(float(diag["latent_std"]))
            step += 1
            if step > 200 and diag["latent_std"] < collapse_std_min:
                raise RuntimeError(f"LMOP collapse detected: latent_std={diag['latent_std']}")
    wall = time.time() - t0
    ckpt = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "width": width,
                "modes": modes,
                "n_layers": n_layers,
                "ema": ema,
                "mask_ratio": mask_ratio,
            },
            "normalize_inputs": normalize_inputs,
            "amendment": "0003_matched_block_jepa",
        },
        ckpt,
    )
    meta = {
        "method": "lmop_jepa",
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "mean_latent_std": float(np.mean(stds)) if stds else None,
        "wall_seconds": wall,
        "parameter_count": count_params(model),
        "shuffle_physics": shuffle_physics,
        "shuffle_mode": shuffle_mode if shuffle_physics else None,
        "normalize_inputs": normalize_inputs,
        "checkpoint": str(ckpt),
        "amendment": "0003_matched_block_jepa",
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    ds.close()
    return meta


def finetune_genuine(
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    *,
    train_indices: list[int],
    init_checkpoint: Path | None,
    init_kind: str,
    epochs: int,
    batch_size: int,
    width: int,
    modes: int,
    n_layers: int,
    lr: float,
    device: torch.device,
    seed: int,
    freeze_epochs: int = 0,
    normalize_inputs: bool = True,
) -> dict[str, Any]:
    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = H5FieldDataset(train_path, indices=train_indices)
    val_ds = H5FieldDataset(val_path)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False, num_workers=0)
    model = DownstreamSolver(width=width, modes=modes, n_layers=n_layers).to(device)
    if init_checkpoint is not None and init_kind not in ("none", "scratch", ""):
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if init_kind == "mml":
            model.load_state_dict(payload["model"])
        elif init_kind == "lmop":
            lmop = LMOPJEPA(width=width, modes=modes, n_layers=n_layers)
            lmop.load_state_dict(payload["model"], strict=True)
            lmop.transfer_to_downstream(model)
        else:
            raise ValueError(f"Unknown init_kind={init_kind}")
    best = float("inf")
    best_path = out_dir / "checkpoint_best.pt"
    history = []
    t0 = time.time()
    if freeze_epochs > 0:
        model.set_backbone_trainable(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    unfroze = freeze_epochs <= 0
    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch >= freeze_epochs and not unfroze:
            model.set_backbone_trainable(True)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            unfroze = True
        model.train()
        tr = []
        for batch in train_loader:
            a, f, u = batch["a"].to(device), batch["f"].to(device), batch["u"].to(device)
            if normalize_inputs:
                a, f = normalize_af(a, f)
            loss = relative_mse(model(a, f), u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr.append(float(loss.item()))
        model.eval()
        va = []
        with torch.no_grad():
            for batch in val_loader:
                a, f, u = batch["a"].to(device), batch["f"].to(device), batch["u"].to(device)
                if normalize_inputs:
                    a, f = normalize_af(a, f)
                va.append(float(relative_mse(model(a, f), u).item()))
        row = {
            "epoch": epoch,
            "train": float(np.mean(tr)),
            "val": float(np.mean(va)),
            "backbone_frozen": bool(freeze_epochs > 0 and epoch < freeze_epochs),
        }
        history.append(row)
        if row["val"] < best:
            best = row["val"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {"width": width, "modes": modes, "n_layers": n_layers},
                    "normalize_inputs": normalize_inputs,
                },
                best_path,
            )
    meta = {
        "best_val": best,
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "wall_seconds": time.time() - t0,
        "parameter_count": count_params(model),
        "init_kind": init_kind,
        "n_train": len(train_ds),
        "normalize_inputs": normalize_inputs,
        "history": history,
        "checkpoint": str(best_path),
        "amendment": "0003_matched_block_jepa",
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    train_ds.close()
    val_ds.close()
    return meta


@torch.no_grad()
def evaluate_model(
    checkpoint: Path,
    data_path: Path,
    out_dir: Path,
    *,
    grid: DarcyGrid,
    width: int,
    modes: int,
    n_layers: int,
    device: torch.device,
    distribution: str,
    normalize_inputs: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DownstreamSolver(width=width, modes=modes, n_layers=n_layers).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    if "normalize_inputs" in payload:
        normalize_inputs = bool(payload["normalize_inputs"])
    ds = H5FieldDataset(data_path)
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        a_raw = item["a"].numpy()
        f_raw = item["f"].numpy()
        u = item["u"].numpy()
        a_t = item["a"].unsqueeze(0).to(device)
        f_t = item["f"].unsqueeze(0).to(device)
        if normalize_inputs:
            a_t, f_t = normalize_af(a_t, f_t)
        pred = model(a_t, f_t).cpu().numpy()[0]
        m = evaluate_example(a_raw, f_raw, u, pred, grid)
        m["index"] = int(item["index"])
        rows.append(m)
    keys = [k for k in rows[0] if k != "index"]
    summary = {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({f"{k}_std": float(np.std([r[k] for r in rows])) for k in keys})
    boot = paired_bootstrap_ci(np.array([r["relative_l2"] for r in rows]))
    result = {
        "distribution": distribution,
        "summary": summary,
        "bootstrap_relative_l2": boot,
        "per_example": rows,
        "normalize_inputs": normalize_inputs,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / "per_example.npz", **{k: np.array([r[k] for r in rows]) for k in keys})
    ds.close()
    return result
