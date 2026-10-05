#!/usr/bin/env python3
"""Freeze recycled Ritz rank and refresh policy from a development-only ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.evaluation.recycled_policy import (
    DevelopmentResidual,
    freeze_recycled_policy,
)


def _load_records(path: Path) -> list[DevelopmentResidual]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("development ledger must be a JSON object")
    if payload.get("source_role") != "development":
        raise ValueError("development ledger must declare source_role=development")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("development ledger must contain a nonempty records list")

    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("record {} must be a JSON object".format(index))
        records.append(
            DevelopmentResidual(
                case_id=str(row["case_id"]),
                rank=int(row["rank"]),
                residual_relative_to_hx=float(row["residual_relative_to_hx"]),
                operator_applications=int(row["operator_applications"]),
                source_role=str(row.get("source_role", payload["source_role"])),
            )
        )
    return records


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

    records = _load_records(args.development_ledger)
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
        "development_ledger": str(args.development_ledger),
        "policy": policy.to_dict(),
        "claim_boundary": (
            "This artifact freezes only the classical recycled-Ritz rank and residual "
            "refresh rule from development-only residuals. It is not a confirmatory "
            "result and does not authorize protected evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
