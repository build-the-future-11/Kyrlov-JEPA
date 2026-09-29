#!/usr/bin/env python3
"""Resolution-scaling study for perturbation-augmented adaptive Ritz solvers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.evaluation.bootstrap import paired_bootstrap_ci
from spectral_krylov_jepa.evaluation.hybrid_spectral import (
    adaptive_ritz,
    coefficient_direction,
    first_order_perturbation_fast,
    first_second_order_perturbation,
    projected_ritz,
    sine_basis,
)
from spectral_krylov_jepa.evaluation.metrics import evaluate_example
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state_from_potential
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.utils.io import write_json


GRID_SIZES = [16, 32, 64]
ID_SEEDS = list(range(810_000, 810_008))
STRONG_SEEDS = list(range(820_000, 820_006))
LOW_SIDE = 3
FULL_SIDE = 7


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fixed_ritz(v: np.ndarray, grid: GridSpec, side: int):
    basis, _ = sine_basis(grid, side)
    return projected_ritz(
        v,
        grid,
        basis,
        assume_orthonormal=True,
    )


def adaptive_from_coeff(v: np.ndarray, grid: GridSpec, coeff: np.ndarray):
    direction = coefficient_direction(
        coeff,
        grid,
        side=FULL_SIDE,
        low_side=LOW_SIDE,
    )
    return adaptive_ritz(
        v,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[direction],
    )


def run_case(
    family: str,
    seed: int,
    grid: GridSpec,
) -> list[dict]:
    v, _ = generate_potential(family, seed, grid=grid)
    gs = solve_ground_state_from_potential(
        v,
        grid,
        tol=1e-9,
        residual_tol=1e-5,
    )

    methods = {}
    for side in [3, 4, 5, 6, 7]:
        methods[f"fixed_ritz_{side * side}"] = fixed_ritz(v, grid, side)

    c1, _ = first_order_perturbation_fast(v, grid, FULL_SIDE)
    methods["pt1_adaptive_10"] = adaptive_from_coeff(v, grid, c1)

    _, c2, _, _ = first_second_order_perturbation(v, grid, FULL_SIDE)
    methods["pt2_adaptive_10"] = adaptive_from_coeff(v, grid, c2)

    rows = []
    for method, result in methods.items():
        m = evaluate_example(
            v,
            gs.wavefunction,
            gs.energy,
            result.wavefunction,
            result.energy,
            grid,
        )
        rows.append(
            {
                "grid": grid.n_interior,
                "family": family,
                "seed": seed,
                "method": method,
                "basis_dim": result.basis_dim,
                **m,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        key = (int(row["grid"]), str(row["family"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    out = []
    for (grid, family, method), items in sorted(groups.items()):
        out.append(
            {
                "grid": grid,
                "family": family,
                "method": method,
                "n": len(items),
                "basis_dim": int(items[0]["basis_dim"]),
                "fidelity_mean": float(np.mean([x["fidelity"] for x in items])),
                "residual_true_e_mean": float(
                    np.mean([x["residual_true_e"] for x in items])
                ),
                "residual_true_e_std": float(
                    np.std([x["residual_true_e"] for x in items])
                ),
                "rel_energy_error_mean": float(
                    np.mean([x["rel_energy_error"] for x in items])
                ),
            }
        )
    return out


def paired(
    rows: list[dict],
    grid: int,
    family: str,
    a: str,
    b: str,
) -> dict:
    aa = {
        int(r["seed"]): float(r["residual_true_e"])
        for r in rows
        if int(r["grid"]) == grid and r["family"] == family and r["method"] == a
    }
    bb = {
        int(r["seed"]): float(r["residual_true_e"])
        for r in rows
        if int(r["grid"]) == grid and r["family"] == family and r["method"] == b
    }
    ids = sorted(set(aa) & set(bb))
    return paired_bootstrap_ci(
        np.asarray([aa[i] for i in ids]),
        np.asarray([bb[i] for i in ids]),
        n_resamples=4000,
        seed=20260929 + grid,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="hybrid_resolution_artifacts")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for n in GRID_SIZES:
        grid = GridSpec(n_interior=n)
        for seed in ID_SEEDS:
            rows.extend(run_case("id_gaussian_mixture", seed, grid))
        for seed in STRONG_SEEDS:
            rows.extend(run_case("ood_strong", seed, grid))

    agg = aggregate(rows)
    comparisons = {}
    for n in GRID_SIZES:
        for family in ["id_gaussian_mixture", "ood_strong"]:
            comparisons[f"{n}_{family}_pt1_vs_fixed25"] = paired(
                rows, n, family, "pt1_adaptive_10", "fixed_ritz_25"
            )
            comparisons[f"{n}_{family}_pt2_vs_fixed49"] = paired(
                rows, n, family, "pt2_adaptive_10", "fixed_ritz_49"
            )

    write_csv(out / "per_example.csv", rows)
    write_csv(out / "aggregate.csv", agg)
    write_json(comparisons, out / "comparisons.json")

    gates = {
        "pt1_beats_fixed25_all_id_resolutions": all(
            comparisons[f"{n}_id_gaussian_mixture_pt1_vs_fixed25"]["ci_high"] < 0
            for n in GRID_SIZES
        ),
        "pt1_beats_fixed25_all_strong_resolutions": all(
            comparisons[f"{n}_ood_strong_pt1_vs_fixed25"]["ci_high"] < 0
            for n in GRID_SIZES
        ),
        "pt2_within_10pct_fixed49_all_id_resolutions": all(
            next(
                x["residual_true_e_mean"]
                for x in agg
                if x["grid"] == n
                and x["family"] == "id_gaussian_mixture"
                and x["method"] == "pt2_adaptive_10"
            )
            <= 1.10
            * next(
                x["residual_true_e_mean"]
                for x in agg
                if x["grid"] == n
                and x["family"] == "id_gaussian_mixture"
                and x["method"] == "fixed_ritz_49"
            )
            for n in GRID_SIZES
        ),
    }

    summary = {
        "status": "ok",
        "grid_sizes": GRID_SIZES,
        "id_seeds": ID_SEEDS,
        "strong_seeds": STRONG_SEEDS,
        "gates": gates,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "claim_boundary": (
            "Resolution scaling uses the same analytic potential seeds sampled on each "
            "grid and a finite synthetic family. It tests mesh robustness, not universal "
            "operator generalization."
        ),
    }
    write_json(summary, out / "summary.json")

    lines = [
        "# Hybrid resolution-scaling study",
        "",
        "| Grid | Family | Method | Basis | Fidelity | Residual | Rel E error |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in agg:
        if row["method"] in {
            "fixed_ritz_9",
            "fixed_ritz_25",
            "fixed_ritz_49",
            "pt1_adaptive_10",
            "pt2_adaptive_10",
        }:
            lines.append(
                f"| {row['grid']} | {row['family']} | {row['method']} | "
                f"{row['basis_dim']} | {row['fidelity_mean']:.8f} | "
                f"{row['residual_true_e_mean']:.6f} | "
                f"{row['rel_energy_error_mean']:.8f} |"
            )
    lines += ["", "## Gates", ""]
    for key, value in gates.items():
        lines.append(f"- {key}: **{value}**")
    lines += ["", "## Claim boundary", "", summary["claim_boundary"], ""]
    (out / "HYBRID_RESOLUTION_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
