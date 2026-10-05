#!/usr/bin/env python3
"""Build and freeze the recycled-Ritz control from development-only residuals.

This script intentionally never solves an exact ground state for a query case.
Exact eigensolves are used only to construct the development recycle space from
a disjoint warm-up set. Query cases expose only the Hamiltonian itself and the
test-time-available normalized Ritz residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.evaluation.recycled_policy import (
    DevelopmentResidual,
    freeze_recycled_policy,
)
from spectral_krylov_jepa.evaluation.recycled_subspace import (
    build_recycle_space,
    solve_recycled_ritz_from_potential,
)
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_potential


CANDIDATE_RANKS = (1, 2, 4, 8)
MAX_OPERATOR_APPLICATIONS = 8
RANKING_QUANTILE = 0.90
THRESHOLD_QUANTILE = 0.95
NEAR_BEST_FACTOR = 1.10
WARMUP_SEEDS = tuple(range(70_000, 70_012))
QUERY_SEEDS = tuple(range(70_100, 70_132))
POTENTIAL_FAMILY = "id_gaussian_mixture"
GRID_INTERIOR = 16


def _canonical_json(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("recycled_ritz_development"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    grid = GridSpec(n_interior=GRID_INTERIOR)

    # Legal recycle information: exact eigenvectors from development warm-up
    # Hamiltonians only. None of the query Hamiltonians are solved exactly.
    warmup_vectors: list[np.ndarray] = []
    for seed in WARMUP_SEEDS:
        potential, _ = generate_potential(POTENTIAL_FAMILY, seed, grid=grid)
        result = solve_ground_state(build_hamiltonian(grid, potential))
        warmup_vectors.append(np.asarray(result.wavefunction, dtype=np.float64).reshape(-1))

    bases = {
        rank: build_recycle_space(
            warmup_vectors,
            n_dof=grid.n_dof,
            rank=rank,
        )
        for rank in CANDIDATE_RANKS
    }

    records: list[dict] = []
    for seed in QUERY_SEEDS:
        potential, _ = generate_potential(POTENTIAL_FAMILY, seed, grid=grid)
        case_id = f"dev-{POTENTIAL_FAMILY}-{seed}"
        for rank in CANDIDATE_RANKS:
            result = solve_recycled_ritz_from_potential(
                potential,
                grid,
                bases[rank],
                refresh_threshold=np.finfo(np.float64).max,
            )
            records.append(
                {
                    "case_id": case_id,
                    "rank": rank,
                    "residual_relative_to_hx": result.residual_relative_to_hx,
                    "residual_norm": result.residual_norm,
                    "operator_applications": result.budget.operator_applications,
                    "reduced_dimension": result.budget.reduced_dimension,
                    "source_role": "development",
                }
            )

    ledger = {
        "schema_version": 1,
        "source_role": "development",
        "potential_family": POTENTIAL_FAMILY,
        "grid_n_interior": GRID_INTERIOR,
        "warmup_seeds": list(WARMUP_SEEDS),
        "query_seeds": list(QUERY_SEEDS),
        "candidate_ranks": list(CANDIDATE_RANKS),
        "max_operator_applications": MAX_OPERATOR_APPLICATIONS,
        "query_exact_eigensolves_performed": False,
        "protected_outcomes_opened": False,
        "records": records,
        "claim_boundary": (
            "Development-only recycled-Ritz residual ledger. Exact eigensolves are used "
            "only for disjoint warm-up basis construction; query cases are not solved "
            "exactly and no protected confirmatory outcome is opened."
        ),
    }
    ledger_bytes = _canonical_json(ledger)
    ledger_path = out / "development_residuals.json"
    ledger_path.write_bytes(ledger_bytes)
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()

    typed_records = [
        DevelopmentResidual(
            case_id=str(row["case_id"]),
            rank=int(row["rank"]),
            residual_relative_to_hx=float(row["residual_relative_to_hx"]),
            operator_applications=int(row["operator_applications"]),
            source_role=str(row["source_role"]),
        )
        for row in records
    ]
    policy = freeze_recycled_policy(
        typed_records,
        candidate_ranks=CANDIDATE_RANKS,
        ranking_quantile=RANKING_QUANTILE,
        threshold_quantile=THRESHOLD_QUANTILE,
        near_best_factor=NEAR_BEST_FACTOR,
        max_operator_applications=MAX_OPERATOR_APPLICATIONS,
    )

    receipt = {
        "status": "frozen-development-policy",
        "protected_outcomes_opened": False,
        "query_exact_eigensolves_performed": False,
        "development_ledger_sha256": ledger_sha256,
        "development_protocol": {
            "potential_family": POTENTIAL_FAMILY,
            "grid_n_interior": GRID_INTERIOR,
            "warmup_seeds": list(WARMUP_SEEDS),
            "query_seeds": list(QUERY_SEEDS),
            "candidate_ranks": list(CANDIDATE_RANKS),
            "max_operator_applications": MAX_OPERATOR_APPLICATIONS,
            "ranking_quantile": RANKING_QUANTILE,
            "threshold_quantile": THRESHOLD_QUANTILE,
            "near_best_factor": NEAR_BEST_FACTOR,
        },
        "policy": policy.to_dict(),
        "claim_boundary": (
            "This freezes only the classical recycled-Ritz development policy. It is not "
            "a protected comparison and does not establish efficacy for recycled Ritz or "
            "Krylov-JEPA."
        ),
    }
    policy_path = out / "frozen_policy.json"
    policy_path.write_bytes(_canonical_json(receipt))

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
