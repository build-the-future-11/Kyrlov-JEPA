#!/usr/bin/env python3
"""Engineering sanity check (NOT claim evidence): does adding coordinate channels let the
downstream FNO fit genuine Darcy solutions?

Trains Scratch only, on the frozen n100 genuine subset, with the frozen optimizer settings, and
reports VALIDATION relative L2 (test splits are not touched). Compares the existing 2-channel
FNO (a_n, f_n) with a 4-channel variant (a_n, f_n, x, y), against the train-mean-field baseline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "lmop_jepa"))

from lmop_jepa.models import FNO2d  # noqa: E402
from lmop_jepa.training import normalize_af, relative_mse  # noqa: E402

DATA = PKG / "data" / "confirmatory_v2"
OUT = PKG / "results" / "tables" / "audit_coord_sanity.json"


class CoordSolver(nn.Module):
    def __init__(self, use_coords: bool, width: int = 32, modes: int = 12, n_layers: int = 4) -> None:
        super().__init__()
        self.use_coords = use_coords
        self.backbone = FNO2d(4 if use_coords else 2, 1, width=width, modes=modes, n_layers=n_layers)

    def forward(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        chans = [a, f]
        if self.use_coords:
            b, h, w = a.shape
            ys = torch.linspace(0, 1, h, device=a.device).view(1, h, 1).expand(b, h, w)
            xs = torch.linspace(0, 1, w, device=a.device).view(1, 1, w).expand(b, h, w)
            chans += [xs, ys]
        return self.backbone(torch.stack(chans, dim=1)).squeeze(1)


def load(name: str, idx=None):
    with h5py.File(DATA / f"{name}.h5", "r") as f:
        a, ff, u = f["a"][:], f["f"][:], f["u"][:]
    if idx is not None:
        a, ff, u = a[idx], ff[idx], u[idx]
    return (torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in (a, ff, u))


def rel_l2(pred: torch.Tensor, u: torch.Tensor) -> float:
    d = (pred - u).reshape(len(u), -1)
    return float((d.norm(dim=1) / u.reshape(len(u), -1).norm(dim=1)).mean())


def train(use_coords: bool, a, f, u, av, fv, uv, epochs: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CoordSolver(use_coords)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    an, fn = normalize_af(a, f)
    avn, fvn = normalize_af(av, fv)
    n = len(a)
    best = float("inf")
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 4):
            j = perm[i : i + 4]
            loss = relative_mse(model(an[j], fn[j]), u[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            best = min(best, rel_l2(model(avn, fvn), uv))
    return {"best_val_rel_l2": best, "wall_seconds": time.time() - t0}


def main() -> int:
    torch.set_num_threads(3)
    manifest = json.loads((PKG / "manifests" / "confirmatory_v2_splits.json").read_text())
    idx = sorted(manifest["subsets"]["n100"])
    a, f, u = load("genuine_train", idx)
    av, fv, uv = load("genuine_val")
    u_bar = u.mean(dim=0, keepdim=True).expand_as(uv)
    report = {
        "evidence_class": "ENGINEERING_SANITY (validation split only; not claim evidence)",
        "subset": "n100",
        "epochs": 60,
        "seed": 11,
        "val_train_mean_field_rel_l2": rel_l2(u_bar, uv),
        "fno_2ch_no_coords": train(False, a, f, u, av, fv, uv, 60, 11),
        "fno_4ch_with_coords": train(True, a, f, u, av, fv, uv, 60, 11),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
