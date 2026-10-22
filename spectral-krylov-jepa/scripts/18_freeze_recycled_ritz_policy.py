#!/usr/bin/env python3
"""Freeze recycled Ritz rank and refresh policy from a development-only ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.evaluation.recycled_policy import (
    DevelopmentResidual,
    development_records_from_payload,
    freeze_recycled_policy,
)


def _load_records(path: Path) -> list[DevelopmentResidual]:
    return development_records_from_payload(json.loads(path.read_bytes()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-ranks", type=int, nargs="+", required=True)
    parser.add_argument("--ranking-quantile", type=float, default=0.90)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--near-best-factor", type=float, default=1.10)
    parser.add_argument("--max-operator-applications", type=int, required=True)
    args = parser.parse_args()

    if args.output.resolve() == args.development_ledger.resolve():
        raise ValueError("output must not overwrite the development ledger")
    # Parse and hash the same read so the receipt cannot bind different bytes.
    ledger_bytes = args.development_ledger.read_bytes()
    records = development_records_from_payload(json.loads(ledger_bytes))
    policy = freeze_recycled_policy(
        records,
        candidate_ranks=args.candidate_ranks,
        ranking_quantile=args.ranking_quantile,
        threshold_quantile=args.threshold_quantile,
        near_best_factor=args.near_best_factor,
        max_operator_applications=args.max_operator_applications,
    )

    payload = {
        "status": "frozen-development-policy",
        "protected_outcomes_opened": False,
        "query_exact_eigensolves_performed": False,
        "development_ledger": str(args.development_ledger),
        "development_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "policy": policy.to_dict(),
        "claim_boundary": (
            "This artifact freezes only the classical recycled-Ritz rank and residual "
            "refresh rule from development-only residuals. It is not a confirmatory "
            "result and does not authorize protected evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A frozen policy is an artifact identity. Rechecks use a fresh destination.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
