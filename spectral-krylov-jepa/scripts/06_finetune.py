#!/usr/bin/env python3
"""Fine-tune downstream ground-state model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune/default.yaml")
    parser.add_argument("--data", type=str, default="experiments/raw/labeled.h5")
    parser.add_argument("--manifest", type=str, default="experiments/manifests/labeled_splits.json")
    parser.add_argument("--subset", type=str, default="n10")
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--method", type=str, default="scratch")
    args = parser.parse_args()
    cfg = load_config(args.config)
    finetune(
        data_path=args.data,
        manifest_path=args.manifest,
        subset=args.subset,
        encoder_path=args.encoder,
        method_name=args.method,
        config=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
