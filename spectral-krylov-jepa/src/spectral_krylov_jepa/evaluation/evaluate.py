"""End-to-end evaluation of fine-tuned checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral_krylov_jepa.data.datasets import labeled_from_manifest
from spectral_krylov_jepa.data.splits import read_manifest
from spectral_krylov_jepa.evaluation.bootstrap import paired_bootstrap_ci
from spectral_krylov_jepa.evaluation.metrics import evaluate_example, summarize_metrics
from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.training.checkpointing import load_checkpoint
from spectral_krylov_jepa.utils.device import get_device
from spectral_krylov_jepa.utils.io import ensure_dir, write_json
from spectral_krylov_jepa.utils.logging import setup_logger


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    data_path: str | Path,
    manifest_path: str | Path,
    split: str = "test_ID",
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    cfg = dict(config or {})
    device = get_device(cfg.get("device", "auto"))
    logger = setup_logger("evaluate")

    manifest = read_manifest(manifest_path)
    ds = labeled_from_manifest(data_path, manifest, split)
    img_size = int(cfg.get("img_size", ds.ny))
    grid = GridSpec(n_interior=img_size)
    ca = cell_area(grid)

    model = DownstreamGroundStateModel(
        img_size=img_size,
        size=cfg.get("model_size", "default"),
        cell_area=ca,
        **cfg.get("encoder_overrides", {}),
    ).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval:{split}"):
            v = batch["potential"].to(device)
            psi = batch["psi0"].numpy()[0]
            e_true = float(batch["energy"].item())
            idx = int(batch["index"].item())
            out = model(v)
            psi_hat = out.psi.cpu().numpy()[0]
            e_hat = float(out.energy.cpu().item())
            metrics = evaluate_example(
                batch["potential"].numpy()[0],
                psi,
                e_true,
                psi_hat,
                e_hat,
                grid,
            )
            metrics["index"] = idx
            rows.append(metrics)
            predictions.append(
                {
                    "index": idx,
                    "energy_true": e_true,
                    "energy_pred": e_hat,
                    "rayleigh_energy": metrics["rayleigh_energy"],
                    "fidelity": metrics["fidelity"],
                    "residual_rel": metrics["residual_rel"],
                    "residual_true_e": metrics["residual_true_e"],
                    "residual_rayleigh": metrics["residual_rayleigh"],
                    "family": batch.get("family", ["?"])[0]
                    if isinstance(batch.get("family"), list)
                    else batch.get("family", "?"),
                }
            )

    summary = summarize_metrics([{k: v for k, v in r.items() if k != "index"} for r in rows])
    fid_vals = np.asarray([r["fidelity"] for r in rows], dtype=np.float64)
    boot = paired_bootstrap_ci(fid_vals, n_resamples=n_bootstrap, seed=int(cfg.get("seed", 0)))

    result = {
        "split": split,
        "n": len(rows),
        "summary": summary,
        "bootstrap_fidelity": boot,
        "per_example": rows,
    }

    if output_dir is not None:
        out = ensure_dir(output_dir)
        write_json(result, out / "metrics.json")
        write_json(predictions, out / "predictions.json")
        logger.info("Wrote evaluation to %s", out)

    ds.close()
    return result
