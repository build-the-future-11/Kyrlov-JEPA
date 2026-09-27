#!/usr/bin/env python3
"""Generate figures from existing result artifacts (never fabricates metrics)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.plotting.learning_curves import plot_label_efficiency
from spectral_krylov_jepa.plotting.ood_figures import plot_ood_comparison
from spectral_krylov_jepa.plotting.ablation_figures import plot_krylov_depth_ablation
from spectral_krylov_jepa.utils.io import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-efficiency-csv", type=str, default="results/tables/label_efficiency.csv")
    parser.add_argument("--ood-json", type=str, default=None)
    parser.add_argument("--ablation-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.label_efficiency_csv)
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["n_labels"] = int(r["n_labels"])
            r["fidelity_mean"] = float(r["fidelity_mean"])
            r["fidelity_std"] = float(r.get("fidelity_std", 0) or 0)
        plot_label_efficiency(rows, out / "label_efficiency.png")
        print(f"Wrote {out / 'label_efficiency.png'}")
    else:
        print(f"No label-efficiency table at {csv_path}; skipping Figure B")

    if args.ood_json and Path(args.ood_json).exists():
        data = read_json(args.ood_json)
        plot_ood_comparison(data, out / "ood_comparison.png")
        print(f"Wrote {out / 'ood_comparison.png'}")

    if args.ablation_json and Path(args.ablation_json).exists():
        data = read_json(args.ablation_json)
        plot_krylov_depth_ablation(data, out / "krylov_depth_ablation.png")
        print(f"Wrote {out / 'krylov_depth_ablation.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
