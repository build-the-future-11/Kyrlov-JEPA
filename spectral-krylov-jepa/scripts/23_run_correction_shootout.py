#!/usr/bin/env python3
"""Fresh correction-vector shootout: perturbation vs Davidson controls."""

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
    residual_expansion_direction,
    sine_basis,
    spectral_davidson_direction,
)
from spectral_krylov_jepa.evaluation.metrics import evaluate_example
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state_from_potential
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.utils.io import write_json


GRIDS = [16, 32, 64]
ID_SEEDS = list(range(910_000, 910_010))
STRONG_SEEDS = list(range(920_000, 920_008))
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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fixed(v: np.ndarray, grid: GridSpec, side: int):
    basis, _ = sine_basis(grid, side)
    return projected_ritz(v, grid, basis, assume_orthonormal=True)


def adaptive_coeff(v: np.ndarray, grid: GridSpec, coeff: np.ndarray):
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


def evaluate_case(family: str, seed: int, grid: GridSpec) -> list[dict]:
    v, _ = generate_potential(family, seed, grid=grid)
    gs = solve_ground_state_from_potential(
        v,
        grid,
        tol=1e-10,
        residual_tol=1e-6,
    )

    methods = {
        "fixed_ritz_9": fixed(v, grid, 3),
        "fixed_ritz_25": fixed(v, grid, 5),
    }

    c1, _ = first_order_perturbation_fast(v, grid, FULL_SIDE)
    methods["pt1_adaptive_10"] = adaptive_coeff(v, grid, c1)

    _, c2, _, _ = first_second_order_perturbation(v, grid, FULL_SIDE)
    methods["pt2_adaptive_10"] = adaptive_coeff(v, grid, c2)

    residual_direction, _ = residual_expansion_direction(
        v,
        grid,
        low_side=LOW_SIDE,
    )
    methods["residual_adaptive_10"] = adaptive_ritz(
        v,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[residual_direction],
    )

    davidson_box, _ = spectral_davidson_direction(
        v,
        grid,
        low_side=LOW_SIDE,
        full_side=FULL_SIDE,
        use_projected_diagonal=False,
    )
    methods["davidson_box_adaptive_10"] = adaptive_ritz(
        v,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[davidson_box],
    )

    davidson_diag, _ = spectral_davidson_direction(
        v,
        grid,
        low_side=LOW_SIDE,
        full_side=FULL_SIDE,
        use_projected_diagonal=True,
    )
    methods["davidson_diag_adaptive_10"] = adaptive_ritz(
        v,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[davidson_diag],
    )

    rows = []
    for method, result in methods.items():
        metrics = evaluate_example(
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
                **metrics,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (int(row["grid"]), str(row["family"]), str(row["method"])),
            [],
        ).append(row)
    out = []
    for (grid, family, method), items in sorted(groups.items()):
        out.append(
            {
                "grid": grid,
                "family": family,
                "method": method,
                "basis_dim": int(items[0]["basis_dim"]),
                "n": len(items),
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


def compare(
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
    seeds = sorted(set(aa) & set(bb))
    return paired_bootstrap_ci(
        np.asarray([aa[s] for s in seeds]),
        np.asarray([bb[s] for s in seeds]),
        n_resamples=5000,
        seed=20260929 + grid,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="correction_shootout_artifacts")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows: list[dict] = []
    for n in GRIDS:
        grid = GridSpec(n_interior=n)
        for seed in ID_SEEDS:
            rows.extend(evaluate_case("id_gaussian_mixture", seed, grid))
        for seed in STRONG_SEEDS:
            rows.extend(evaluate_case("ood_strong", seed, grid))

    agg = aggregate(rows)
    comparisons = {}
    for n in GRIDS:
        for family in ["id_gaussian_mixture", "ood_strong"]:
            for competitor in [
                "fixed_ritz_25",
                "residual_adaptive_10",
                "davidson_box_adaptive_10",
                "davidson_diag_adaptive_10",
            ]:
                comparisons[
                    f"{n}_{family}_pt1_vs_{competitor}"
                ] = compare(
                    rows,
                    n,
                    family,
                    "pt1_adaptive_10",
                    competitor,
                )

    write_csv(out / "per_example.csv", rows)
    write_csv(out / "aggregate.csv", agg)
    write_json(comparisons, out / "comparisons.json")

    gates = {
        "pt1_beats_fixed25_all_grids_id": all(
            comparisons[f"{n}_id_gaussian_mixture_pt1_vs_fixed_ritz_25"]["ci_high"] < 0
            for n in GRIDS
        ),
        "pt1_beats_residual_control_all_grids_id": all(
            comparisons[
                f"{n}_id_gaussian_mixture_pt1_vs_residual_adaptive_10"
            ]["ci_high"]
            < 0
            for n in GRIDS
        ),
        "pt1_beats_davidson_box_all_grids_id": all(
            comparisons[
                f"{n}_id_gaussian_mixture_pt1_vs_davidson_box_adaptive_10"
            ]["ci_high"]
            < 0
            for n in GRIDS
        ),
        "pt1_beats_davidson_diag_all_grids_id": all(
            comparisons[
                f"{n}_id_gaussian_mixture_pt1_vs_davidson_diag_adaptive_10"
            ]["ci_high"]
            < 0
            for n in GRIDS
        ),
    }

    summary = {
        "status": "ok",
        "grid_sizes": GRIDS,
        "id_seeds": ID_SEEDS,
        "strong_seeds": STRONG_SEEDS,
        "gates": gates,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "claim_boundary": (
            "This fresh shootout distinguishes the PT1 correction from classical "
            "unpreconditioned and Davidson-style single-vector subspace expansions "
            "on the same finite synthetic potential families."
        ),
    }
    write_json(summary, out / "summary.json")

    lines = [
        "# Fresh correction-vector shootout",
        "",
        "| Grid | Family | Method | Basis | Fidelity | Residual | Rel E error |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in agg:
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
    (out / "CORRECTION_SHOOTOUT_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
