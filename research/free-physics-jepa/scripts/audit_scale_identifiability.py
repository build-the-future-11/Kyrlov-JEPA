#!/usr/bin/env python3
"""Audit: is the target scale identifiable from the amendment-0003 normalized inputs?

With a_n = a / mean(a) and f_n = f / rms(f), linearity of the Darcy operator gives
A_a = mean(a) * A_{a_n}, so for genuine data (f ≡ 1)

    u * mean(a) = A_{a_n}^{-1} 1

is an exact function of the model inputs, while mean(a) itself is removed. A model that
outputs physical u therefore faces an unobserved per-example scale 1/mean(a).

This script measures (1) how much mean(a) varies, (2) the relative-L2 floor of an oracle that
knows the exact normalized solution u*mean(a) but must pick one global scale (fit on train), and
(3) how per-example errors of trained models correlate with that oracle error.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "data" / "confirmatory_v2"
RUNS = PKG / "runs"
OUT = PKG / "results" / "tables" / "audit_scale_identifiability.json"


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(DATA / f"{name}.h5", "r") as f:
        return (
            f["a"][:].astype(np.float64),
            f["f"][:].astype(np.float64),
            f["u"][:].astype(np.float64),
        )


def oracle_scale(m_train: np.ndarray) -> float:
    # Minimize mean |c*m - 1| over c: weighted median of 1/m with weights m.
    inv = 1.0 / m_train
    order = np.argsort(inv)
    w = m_train[order] / m_train.sum()
    return float(inv[order][np.searchsorted(np.cumsum(w), 0.5)])


def per_example_errors(run: str, tag: str, dist: str) -> np.ndarray | None:
    path = RUNS / run / "eval" / f"{tag}_{dist}" / "metrics.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["per_example"]
    return np.array([r["relative_l2"] for r in rows])


def main() -> int:
    a_tr, _, _ = load("genuine_train")
    c = oracle_scale(a_tr.mean(axis=(1, 2)))
    report: dict = {"oracle_global_scale": c, "splits": {}}
    probe_run = "confirmatory_s3_v2_20260926T185418Z"
    s2_run = "confirmatory_s2_v2_20260926T162730Z"
    for name, dist in (("genuine_test_id", "id"), ("genuine_ood", "ood")):
        a, f, _ = load(name)
        m = a.mean(axis=(1, 2))
        oracle = np.abs(c * m - 1.0)
        split = {
            "forcing_constant": bool(np.allclose(f, f.reshape(len(f), -1)[:, :1, None])),
            "mean_a_mean": float(m.mean()),
            "mean_a_cv": float(m.std() / m.mean()),
            "mean_a_min": float(m.min()),
            "mean_a_max": float(m.max()),
            "oracle_rel_l2_mean": float(oracle.mean()),
            "oracle_rel_l2_median": float(np.median(oracle)),
            "model_correlation_with_oracle_error": {},
        }
        for run, tag in (
            (probe_run, "probe_scratch_n100_s11"),
            (probe_run, "probe_mml_direct_n100_s11"),
            (probe_run, "probe_lmop_jepa_n100_s11"),
            (s2_run, "scratch_n100_s11"),
            (s2_run, "mml_direct_n100_s11"),
        ):
            err = per_example_errors(run, tag, dist)
            if err is None or len(err) != len(oracle):
                continue
            split["model_correlation_with_oracle_error"][f"{run}/{tag}"] = {
                "model_rel_l2_mean": float(err.mean()),
                "pearson_r": float(np.corrcoef(err, oracle)[0, 1]),
            }
        report["splits"][dist] = split
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
