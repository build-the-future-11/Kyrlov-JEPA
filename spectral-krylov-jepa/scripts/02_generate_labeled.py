#!/usr/bin/env python3
"""Generate labeled eigenstate data and freeze split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/raw/labeled.h5")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test-id", type=int, default=40)
    parser.add_argument("--n-ood-each", type=int, default=40)
    parser.add_argument("--n-interior", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--manifest-name", type=str, default="labeled_splits")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    grid = GridSpec(n_interior=int(cfg.get("n_interior", args.n_interior)))
    generate_labeled_dataset(
        output_path=cfg.get("output", args.output),
        grid=grid,
        n_train=int(cfg.get("n_train", args.n_train)),
        n_val=int(cfg.get("n_val", args.n_val)),
        n_test_id=int(cfg.get("n_test_id", args.n_test_id)),
        n_ood_each=int(cfg.get("n_ood_each", args.n_ood_each)),
        subset_sizes=cfg.get("subset_sizes", [10, 25, 50, 100]),
        split_seed=int(cfg.get("split_seed", args.split_seed)),
        manifest_name=cfg.get("manifest_name", args.manifest_name),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
