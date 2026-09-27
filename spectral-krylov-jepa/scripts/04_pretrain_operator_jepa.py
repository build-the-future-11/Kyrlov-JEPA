#!/usr/bin/env python3
"""Pretrain Operator-JEPA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain/operator_jepa.yaml")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = args.data or cfg.get("data_path", "experiments/raw/unlabeled.h5")
    pretrain(method="operator", data_path=data, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
