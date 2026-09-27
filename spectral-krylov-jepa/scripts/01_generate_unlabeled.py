#!/usr/bin/env python3
"""Generate unlabeled Krylov pretraining data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/raw/unlabeled.h5")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-interior", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--shuffle-physics", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    n_examples = int(cfg.get("n_examples", args.n_examples))
    out = cfg.get("output", args.output)
    grid = GridSpec(n_interior=int(cfg.get("n_interior", args.n_interior)))

    generate_unlabeled_dataset(
        output_path=out,
        n_examples=n_examples,
        n_starts=int(cfg.get("n_starts", args.n_starts)),
        depth=int(cfg.get("depth", args.depth)),
        grid=grid,
        base_seed=int(cfg.get("base_seed", args.seed)),
        shuffle_physics=bool(cfg.get("shuffle_physics", args.shuffle_physics)),
        manifest_path=str(Path(out).with_suffix(".manifest.json")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
