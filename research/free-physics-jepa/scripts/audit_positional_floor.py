#!/usr/bin/env python3
"""Audit: can the downstream FNO represent the Dirichlet solution shape at all?

`FNO2d` has no coordinate channels and no padding, so it is exactly equivariant to periodic
translations. For genuine data f ≡ 1 (constant after normalization), so the only input is a
stationary random field a_n; the model cannot locate the zero-Dirichlet boundary, whereas u is a
boundary-pinned bump. This script compares trained models against trivial baselines:

  - zero field                        (rel L2 = 1)
  - per-example best constant c_i     (floor for any spatially constant output)
  - train-mean field ū(x)             (position-aware, input-blind)
  - ū(x) rescaled per example by 1/mean(a)  (position-aware, uses only mean(a))

and measures how spatially flat the trained models' predictions are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "lmop_jepa"))

from lmop_jepa.models import DownstreamSolver  # noqa: E402
from lmop_jepa.training import normalize_af  # noqa: E402

DATA = PKG / "data" / "confirmatory_v2"
RUNS = PKG / "runs"
OUT = PKG / "results" / "tables" / "audit_positional_floor.json"


def load(name: str):
    with h5py.File(DATA / f"{name}.h5", "r") as f:
        return f["a"][:].astype(np.float64), f["f"][:].astype(np.float64), f["u"][:].astype(np.float64)


def rel_l2(pred: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = (pred - u).reshape(len(u), -1)
    return np.linalg.norm(d, axis=1) / np.linalg.norm(u.reshape(len(u), -1), axis=1)


def flatness(x: np.ndarray) -> np.ndarray:
    """Spatial std / |spatial mean| per example (0 = perfectly flat)."""
    flat = x.reshape(len(x), -1)
    return flat.std(axis=1) / (np.abs(flat.mean(axis=1)) + 1e-12)


@torch.no_grad()
def predict(ckpt: Path, a: np.ndarray, f: np.ndarray) -> np.ndarray:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {"width": 32, "modes": 12, "n_layers": 4})
    model = DownstreamSolver(width=cfg["width"], modes=cfg["modes"], n_layers=cfg["n_layers"])
    model.load_state_dict(payload["model"])
    model.eval()
    at = torch.from_numpy(a).float()
    ft = torch.from_numpy(f).float()
    if payload.get("normalize_inputs", True):
        at, ft = normalize_af(at, ft)
    return model(at, ft).numpy().astype(np.float64)


def main() -> int:
    torch.set_num_threads(2)
    a_tr, _, u_tr = load("genuine_train")
    u_bar = u_tr.mean(axis=0)
    m_tr = a_tr.mean(axis=(1, 2))
    u_bar_scaled_ref = (u_tr * m_tr[:, None, None]).mean(axis=0)

    ckpts = {
        "probe_scratch_n100_s11": RUNS / "confirmatory_s3_v2_20260926T185418Z/finetune/probe_scratch_n100_s11/checkpoint_best.pt",
        "probe_mml_direct_n100_s11": RUNS / "confirmatory_s3_v2_20260926T185418Z/finetune/probe_mml_direct_n100_s11/checkpoint_best.pt",
        "probe_lmop_jepa_n100_s11": RUNS / "confirmatory_s3_v2_20260926T185418Z/finetune/probe_lmop_jepa_n100_s11/checkpoint_best.pt",
        "s2_full_ft_scratch_n100_s11": RUNS / "confirmatory_s2_v2_20260926T162730Z/finetune/scratch_n100_s11/checkpoint_best.pt",
        "s2_full_ft_mml_direct_n100_s11": RUNS / "confirmatory_s2_v2_20260926T162730Z/finetune/mml_direct_n100_s11/checkpoint_best.pt",
    }

    report: dict = {"splits": {}}
    for name, dist in (("genuine_test_id", "id"), ("genuine_ood", "ood")):
        a, f, u = load(name)
        m = a.mean(axis=(1, 2))
        const = np.broadcast_to(u.reshape(len(u), -1).mean(axis=1)[:, None, None], u.shape)
        split = {
            "truth_flatness_mean": float(flatness(u).mean()),
            "baselines_rel_l2_mean": {
                "zero": 1.0,
                "per_example_best_constant": float(rel_l2(const, u).mean()),
                "train_mean_field": float(rel_l2(np.broadcast_to(u_bar, u.shape), u).mean()),
                "train_mean_field_over_mean_a": float(rel_l2(u_bar_scaled_ref[None] / m[:, None, None], u).mean()),
            },
            "models": {},
        }
        for tag, ckpt in ckpts.items():
            if not ckpt.exists():
                continue
            pred = predict(ckpt, a, f)
            split["models"][tag] = {
                "rel_l2_mean": float(rel_l2(pred, u).mean()),
                "prediction_flatness_mean": float(flatness(pred).mean()),
                "boundary_row_abs_mean_over_center_abs": float(
                    np.abs(np.concatenate([pred[:, 0, :], pred[:, -1, :], pred[:, :, 0], pred[:, :, -1]], axis=1)).mean()
                    / (np.abs(pred[:, 28:36, 28:36]).mean() + 1e-12)
                ),
            }
        split["truth_boundary_row_abs_mean_over_center_abs"] = float(
            np.abs(np.concatenate([u[:, 0, :], u[:, -1, :], u[:, :, 0], u[:, :, -1]], axis=1)).mean()
            / (np.abs(u[:, 28:36, 28:36]).mean() + 1e-12)
        )
        report["splits"][dist] = split
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
