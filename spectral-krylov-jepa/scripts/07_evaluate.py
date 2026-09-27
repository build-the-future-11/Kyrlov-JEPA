#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="experiments/raw/labeled.h5")
    parser.add_argument("--manifest", type=str, default="experiments/manifests/labeled_splits.json")
    parser.add_argument("--split", type=str, default="test_ID")
    parser.add_argument("--config", type=str, default="configs/eval/default.yaml")
    parser.add_argument("--output", type=str, default="results/metrics")
    args = parser.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_path=args.data,
        manifest_path=args.manifest,
        split=args.split,
        config=cfg,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
