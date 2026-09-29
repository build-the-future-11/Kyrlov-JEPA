#!/usr/bin/env python3
"""Fresh confirmatory study for adaptive perturbation--Ritz Krylov hybrids.

The primary question is whether one learned high-frequency direction, proposed
from frozen Krylov-JEPA features, can make a 10-dimensional adaptive
Rayleigh--Ritz subspace compete with a much larger fixed sine basis.

No final-test metric is used for hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.bootstrap import paired_bootstrap_ci
from spectral_krylov_jepa.evaluation.hybrid_spectral import (
    adaptive_ritz,
    coefficient_direction,
    first_order_perturbation_fast,
    first_second_order_perturbation,
    high_mode_target,
    low_mode_mask,
    physics_features,
    projected_ritz,
    sine_basis,
    spectral_coefficients,
)
from spectral_krylov_jepa.evaluation.metrics import evaluate_example, schrodinger_residual
from spectral_krylov_jepa.models.encoders import PotentialEncoder, default_encoder_kwargs
from spectral_krylov_jepa.physics.eigensolver import solve_ground_state_from_potential
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.potentials import generate_potential
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.training.seed import seed_everything
from spectral_krylov_jepa.utils.io import write_json


PRETRAIN_SEEDS = [11, 23, 47]
LABEL_BUDGETS = [5, 10, 20, 40]
RIDGE_ALPHAS = [1e-5, 1e-3, 1e-1, 10.0]
LOW_SIDE = 3
FULL_SIDE = 7
FRESH_SEEDS = {
    "unlabeled": 610_000,
    "train": 630_000,
    "validation": 650_000,
    "test_id": 670_000,
    "ood": 690_000,
}
PRIMARY_LABEL_BUDGET = 20


@dataclass(frozen=True)
class Sample:
    potential: np.ndarray
    psi: np.ndarray
    energy: float
    family: str
    seed: int


@dataclass
class RidgeMap:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> "RidgeMap":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        xs = (x - mean) / scale
        ym = y.mean(axis=0)
        yc = y - ym
        if xs.shape[0] <= xs.shape[1]:
            dual = np.linalg.solve(
                xs @ xs.T + alpha * np.eye(xs.shape[0]),
                yc,
            )
            coef = xs.T @ dual
        else:
            coef = np.linalg.solve(
                xs.T @ xs + alpha * np.eye(xs.shape[1]),
                xs.T @ yc,
            )
        return cls(mean=mean, scale=scale, coef=coef, intercept=ym, alpha=float(alpha))

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return xs @ self.coef + self.intercept


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def solve_sample(potential: np.ndarray, grid: GridSpec, family: str, seed: int) -> Sample:
    gs = solve_ground_state_from_potential(
        potential,
        grid,
        tol=1e-10,
        residual_tol=1e-6,
    )
    return Sample(
        potential=np.asarray(potential, dtype=np.float64),
        psi=gs.wavefunction,
        energy=float(gs.energy),
        family=family,
        seed=seed,
    )


def generate_standard_samples(
    family: str,
    n: int,
    seed0: int,
    grid: GridSpec,
) -> list[Sample]:
    out: list[Sample] = []
    for i in range(n):
        seed = seed0 + i
        v, _ = generate_potential(family, seed, grid=grid)
        out.append(solve_sample(v, grid, family, seed))
    return out


def generate_hard_tunnel(seed: int, grid: GridSpec) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = grid.meshgrid()
    cy = float(rng.uniform(0.42, 0.58))
    sep = float(rng.uniform(0.42, 0.58))
    cx1 = 0.5 - sep / 2
    cx2 = 0.5 + sep / 2
    amp = float(rng.uniform(-24.0, -16.0))
    sigma = float(rng.uniform(0.045, 0.075))
    barrier_amp = float(rng.uniform(5.0, 11.0))
    barrier_sigma = float(rng.uniform(0.025, 0.05))
    v = amp * np.exp(-((x - cx1) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    v += amp * np.exp(-((x - cx2) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    v += barrier_amp * np.exp(-((x - 0.5) ** 2) / (2 * barrier_sigma**2))
    return v


def generate_rough_multiscale(seed: int, grid: GridSpec) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = grid.meshgrid()
    cx = float(rng.uniform(0.35, 0.65))
    cy = float(rng.uniform(0.35, 0.65))
    v = -8.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.22**2))
    sx = float(rng.uniform(0.18, 0.82))
    sy = float(rng.uniform(0.18, 0.82))
    v += -13.0 * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 0.04**2))
    for _ in range(4):
        kx = int(rng.integers(3, 8))
        ky = int(rng.integers(3, 8))
        phase = float(rng.uniform(0, 2 * np.pi))
        v += float(rng.uniform(-1.5, 1.5)) * np.sin(2 * np.pi * kx * x + phase) * np.sin(
            2 * np.pi * ky * y
        )
    return v


def generate_custom_samples(
    name: str,
    fn: Callable[[int, GridSpec], np.ndarray],
    n: int,
    seed0: int,
    grid: GridSpec,
) -> list[Sample]:
    return [
        solve_sample(fn(seed0 + i, grid), grid, name, seed0 + i)
        for i in range(n)
    ]


def encoder_from_path(path: str | Path | None, seed: int, grid: GridSpec) -> PotentialEncoder:
    seed_everything(seed)
    kwargs = default_encoder_kwargs("smoke")
    enc = PotentialEncoder(img_size=grid.n_interior, **kwargs)
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        enc.load_state_dict(payload["encoder"], strict=True)
    enc.eval()
    return enc


@torch.no_grad()
def encoder_features(enc: PotentialEncoder, samples: list[Sample]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(samples), 16):
        batch = np.stack([s.potential for s in samples[start : start + 16]])
        z, _ = enc(torch.from_numpy(batch).float())
        rows.append(z.numpy())
    return np.concatenate(rows, axis=0).astype(np.float64)


def physics_matrix(samples: list[Sample], grid: GridSpec) -> np.ndarray:
    return np.stack(
        [physics_features(s.potential, grid, side=FULL_SIDE) for s in samples],
        axis=0,
    )


def target_matrix(samples: list[Sample], grid: GridSpec) -> np.ndarray:
    return np.stack(
        [
            high_mode_target(
                s.psi,
                grid,
                side=FULL_SIDE,
                low_side=LOW_SIDE,
            )
            for s in samples
        ],
        axis=0,
    )


def pt1_matrix(samples: list[Sample], grid: GridSpec) -> np.ndarray:
    mask = low_mode_mask(FULL_SIDE, LOW_SIDE)
    rows = []
    for sample in samples:
        c1, _ = first_order_perturbation_fast(
            sample.potential,
            grid,
            FULL_SIDE,
        )
        c1 = c1.copy()
        c1[mask] = 0.0
        rows.append(c1)
    return np.stack(rows, axis=0)


def pt2_matrix(samples: list[Sample], grid: GridSpec) -> np.ndarray:
    mask = low_mode_mask(FULL_SIDE, LOW_SIDE)
    rows = []
    for s in samples:
        _, c2, _, _ = first_second_order_perturbation(
            s.potential,
            grid,
            FULL_SIDE,
        )
        c2 = c2.copy()
        c2[mask] = 0.0
        rows.append(c2)
    return np.stack(rows, axis=0)


def fixed_ritz(sample: Sample, grid: GridSpec, side: int):
    basis, _ = sine_basis(grid, side)
    return projected_ritz(sample.potential, grid, basis)


def adaptive_from_coeff(
    sample: Sample,
    grid: GridSpec,
    coeff: np.ndarray,
):
    direction = coefficient_direction(
        coeff,
        grid,
        side=FULL_SIDE,
        low_side=LOW_SIDE,
    )
    return adaptive_ritz(
        sample.potential,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[direction],
    )


def adaptive_two_direction(
    sample: Sample,
    grid: GridSpec,
    first: np.ndarray,
    second: np.ndarray,
):
    d1 = coefficient_direction(
        first,
        grid,
        side=FULL_SIDE,
        low_side=LOW_SIDE,
    )
    d2 = coefficient_direction(
        second,
        grid,
        side=FULL_SIDE,
        low_side=LOW_SIDE,
    )
    return adaptive_ritz(
        sample.potential,
        grid,
        low_side=LOW_SIDE,
        proposal_vectors=[d1, d2],
    )


def metric_row(
    method: str,
    sample: Sample,
    result,
    grid: GridSpec,
    *,
    seed: int | None,
    n_labels: int,
    split: str,
) -> dict:
    metrics = evaluate_example(
        sample.potential,
        sample.psi,
        sample.energy,
        result.wavefunction,
        result.energy,
        grid,
    )
    return {
        "method": method,
        "seed": seed if seed is not None else -1,
        "n_labels": n_labels,
        "split": split,
        "sample_seed": sample.seed,
        "basis_dim": int(result.basis_dim),
        "proposal_rank": int(result.proposal_rank),
        **metrics,
    }


def validation_score(
    model: RidgeMap,
    features: np.ndarray,
    samples: list[Sample],
    grid: GridSpec,
    *,
    pt2: np.ndarray | None = None,
) -> float:
    pred = model.predict(features)
    vals = []
    for i, sample in enumerate(samples):
        coeff = pred[i] if pt2 is None else pt2[i] + pred[i]
        result = adaptive_from_coeff(sample, grid, coeff)
        vals.append(
            schrodinger_residual(
                sample.potential,
                result.wavefunction,
                sample.energy,
                grid,
            )
        )
    return float(np.mean(vals))


def choose_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val_samples: list[Sample],
    grid: GridSpec,
    *,
    pt2_val: np.ndarray | None = None,
) -> tuple[RidgeMap, dict]:
    candidates: list[tuple[float, float, RidgeMap]] = []
    for alpha in RIDGE_ALPHAS:
        model = RidgeMap.fit(x_train, y_train, alpha)
        score = validation_score(
            model,
            x_val,
            val_samples,
            grid,
            pt2=pt2_val,
        )
        candidates.append((score, alpha, model))
    candidates.sort(key=lambda x: x[0])
    best_score, best_alpha, best_model = candidates[0]
    return best_model, {
        "alpha": float(best_alpha),
        "validation_residual": float(best_score),
        "candidate_scores": [
            {"alpha": float(a), "residual": float(s)}
            for s, a, _ in candidates
        ],
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["n_labels"]), str(row["split"]))
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for (method, n_labels, split), items in sorted(groups.items()):
        seeds = sorted(set(int(x["seed"]) for x in items))
        sample_seeds = sorted(set(int(x["sample_seed"]) for x in items))
        seed_means = []
        for seed in seeds:
            subset = [x for x in items if int(x["seed"]) == seed]
            seed_means.append(
                float(np.mean([x["residual_true_e"] for x in subset]))
            )
        out.append(
            {
                "method": method,
                "n_labels": n_labels,
                "split": split,
                "n_rows": len(items),
                "n_model_seeds": len(seeds),
                "n_samples": len(sample_seeds),
                "basis_dim_mean": float(np.mean([x["basis_dim"] for x in items])),
                "fidelity_mean": float(np.mean([x["fidelity"] for x in items])),
                "fidelity_std": float(np.std([x["fidelity"] for x in items])),
                "residual_true_e_mean": float(
                    np.mean([x["residual_true_e"] for x in items])
                ),
                "residual_true_e_std": float(
                    np.std([x["residual_true_e"] for x in items])
                ),
                "residual_seed_std": float(np.std(seed_means)) if seed_means else 0.0,
                "rel_energy_error_mean": float(
                    np.mean([x["rel_energy_error"] for x in items])
                ),
            }
        )
    return out


def seed_averaged_per_example(
    rows: list[dict],
    method: str,
    n_labels: int,
    split: str,
    metric: str,
) -> dict[int, float]:
    subset = [
        r
        for r in rows
        if r["method"] == method
        and int(r["n_labels"]) == n_labels
        and r["split"] == split
    ]
    grouped: dict[int, list[float]] = {}
    for row in subset:
        grouped.setdefault(int(row["sample_seed"]), []).append(float(row[metric]))
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def bootstrap_comparison(
    rows: list[dict],
    method_a: str,
    method_b: str,
    n_labels: int,
    split: str,
    metric: str,
    *,
    n_labels_b: int | None = None,
) -> dict:
    a = seed_averaged_per_example(rows, method_a, n_labels, split, metric)
    b = seed_averaged_per_example(
        rows,
        method_b,
        n_labels if n_labels_b is None else n_labels_b,
        split,
        metric,
    )
    common = sorted(set(a) & set(b))
    return paired_bootstrap_ci(
        np.asarray([a[i] for i in common]),
        np.asarray([b[i] for i in common]),
        n_resamples=3000,
        seed=2026,
    )


def krylov_subspace_result(
    sample: Sample,
    grid: GridSpec,
    q0: np.ndarray,
    depth: int,
):
    ham = build_hamiltonian(grid, sample.potential)
    q = np.asarray(q0, dtype=np.float64).reshape(-1)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    cols = [q]
    current = q
    for _ in range(depth):
        v = ham.matvec(current)
        for old in cols:
            v -= float(np.dot(old, v)) * old
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-10:
            break
        current = v / nrm
        cols.append(current)
    return projected_ritz(
        sample.potential,
        grid,
        np.stack(cols, axis=1),
    )


def convergence_curve(
    samples: list[Sample],
    grid: GridSpec,
    starts: list[np.ndarray],
    max_depth: int = 6,
) -> list[float]:
    means = []
    for depth in range(max_depth + 1):
        vals = []
        for sample, q0 in zip(samples, starts):
            result = krylov_subspace_result(sample, grid, q0, depth)
            vals.append(
                schrodinger_residual(
                    sample.potential,
                    result.wavefunction,
                    sample.energy,
                    grid,
                )
            )
        means.append(float(np.mean(vals)))
    return means


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="hybrid_breakthrough_artifacts")
    parser.add_argument("--pretrain-steps", type=int, default=140)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    grid = GridSpec(n_interior=16)

    train = generate_standard_samples(
        "id_gaussian_mixture", 80, FRESH_SEEDS["train"], grid
    )
    validation = generate_standard_samples(
        "id_gaussian_mixture", 20, FRESH_SEEDS["validation"], grid
    )
    test_id = generate_standard_samples(
        "id_gaussian_mixture", 30, FRESH_SEEDS["test_id"], grid
    )
    splits: dict[str, list[Sample]] = {
        "test_ID": test_id,
        "test_OOD_narrow": generate_standard_samples(
            "ood_narrow", 16, FRESH_SEEDS["ood"], grid
        ),
        "test_OOD_strong": generate_standard_samples(
            "ood_strong", 16, FRESH_SEEDS["ood"] + 10_000, grid
        ),
        "test_OOD_double": generate_standard_samples(
            "ood_double", 16, FRESH_SEEDS["ood"] + 20_000, grid
        ),
        "test_OOD_rough": generate_standard_samples(
            "ood_rough", 16, FRESH_SEEDS["ood"] + 30_000, grid
        ),
        "test_hard_tunnel": generate_custom_samples(
            "hard_tunnel",
            generate_hard_tunnel,
            16,
            FRESH_SEEDS["ood"] + 40_000,
            grid,
        ),
        "test_rough_multiscale": generate_custom_samples(
            "rough_multiscale",
            generate_rough_multiscale,
            16,
            FRESH_SEEDS["ood"] + 50_000,
            grid,
        ),
    }

    unlabeled_path = out / "unlabeled.h5"
    generate_unlabeled_dataset(
        output_path=unlabeled_path,
        n_examples=260,
        n_starts=1,
        depth=3,
        grid=grid,
        base_seed=FRESH_SEEDS["unlabeled"],
        manifest_path=out / "unlabeled.manifest.json",
    )

    all_for_features = train + validation + [
        s for split in splits.values() for s in split
    ]
    physics_all = physics_matrix(all_for_features, grid)
    n_train = len(train)
    n_val = len(validation)
    physics_train = physics_all[:n_train]
    physics_val = physics_all[n_train : n_train + n_val]
    physics_split: dict[str, np.ndarray] = {}
    cursor = n_train + n_val
    for name, samples in splits.items():
        physics_split[name] = physics_all[cursor : cursor + len(samples)]
        cursor += len(samples)

    target_train = target_matrix(train, grid)
    target_val = target_matrix(validation, grid)
    pt2_train = pt2_matrix(train, grid)
    pt2_val = pt2_matrix(validation, grid)

    rows: list[dict] = []
    tuning_rows: list[dict] = []
    encoder_diag: list[dict] = []

    for split_name, samples in splits.items():
        for sample in samples:
            for side in [3, 4, 5, 7]:
                fixed = fixed_ritz(sample, grid, side)
                rows.append(
                    metric_row(
                        f"fixed_ritz_{side * side}",
                        sample,
                        fixed,
                        grid,
                        seed=None,
                        n_labels=0,
                        split=split_name,
                    )
                )

            c1, _ = first_order_perturbation_fast(
                sample.potential,
                grid,
                FULL_SIDE,
            )
            pt1 = adaptive_from_coeff(sample, grid, c1)
            rows.append(
                metric_row(
                    "pt1_adaptive_10",
                    sample,
                    pt1,
                    grid,
                    seed=None,
                    n_labels=0,
                    split=split_name,
                )
            )

            _, c2, _, _ = first_second_order_perturbation(
                sample.potential,
                grid,
                FULL_SIDE,
            )
            pt2 = adaptive_from_coeff(sample, grid, c2)
            rows.append(
                metric_row(
                    "pt2_adaptive_10",
                    sample,
                    pt2,
                    grid,
                    seed=None,
                    n_labels=0,
                    split=split_name,
                )
            )

            oracle_coeff = high_mode_target(
                sample.psi,
                grid,
                side=FULL_SIDE,
                low_side=LOW_SIDE,
            )
            oracle = adaptive_from_coeff(sample, grid, oracle_coeff)
            rows.append(
                metric_row(
                    "oracle_adaptive_10",
                    sample,
                    oracle,
                    grid,
                    seed=None,
                    n_labels=-1,
                    split=split_name,
                )
            )

    encoder_feature_sets: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for pretrain_seed in PRETRAIN_SEEDS:
        pretrain_cfg = {
            "seed": pretrain_seed,
            "device": "cpu",
            "model_size": "smoke",
            "img_size": 16,
            "batch_size": 4,
            "max_steps": int(args.pretrain_steps),
            "lr": 3e-4,
            "context_steps": 2,
            "num_workers": 0,
            "log_every": max(20, int(args.pretrain_steps) // 2),
        }
        legacy = pretrain(
            method="krylov",
            data_path=unlabeled_path,
            run_dir=out / f"seed_{pretrain_seed}" / "pretrain_krylov",
            config=pretrain_cfg,
        )
        projected = pretrain(
            method="krylov",
            data_path=unlabeled_path,
            run_dir=out / f"seed_{pretrain_seed}" / "pretrain_projected",
            config={
                **pretrain_cfg,
                "lambda_projected_ritz": 1.0,
                "projected_modes": 9,
            },
        )

        encoders = {
            "scratch": encoder_from_path(None, pretrain_seed, grid),
            "krylov": encoder_from_path(legacy["encoder_path"], pretrain_seed, grid),
            "projected": encoder_from_path(projected["encoder_path"], pretrain_seed, grid),
        }

        seed_features: dict[str, dict[str, np.ndarray]] = {}
        for enc_name, enc in encoders.items():
            z_all = encoder_features(enc, all_for_features)
            z_train = z_all[:n_train]
            centered = z_train - z_train.mean(axis=0)
            singular = np.linalg.svd(centered, compute_uv=False)
            power = singular * singular
            prob = power / max(float(power.sum()), 1e-30)
            erank = float(
                np.exp(-np.sum(prob * np.log(np.clip(prob, 1e-30, None))))
            )
            encoder_diag.append(
                {
                    "seed": pretrain_seed,
                    "encoder": enc_name,
                    "effective_rank": erank,
                    "mean_feature_std": float(z_train.std(axis=0).mean()),
                }
            )
            split_map: dict[str, np.ndarray] = {
                "train": z_train,
                "validation": z_all[n_train : n_train + n_val],
            }
            pos = n_train + n_val
            for split_name, samples in splits.items():
                split_map[split_name] = z_all[pos : pos + len(samples)]
                pos += len(samples)
            seed_features[enc_name] = split_map
        encoder_feature_sets[pretrain_seed] = seed_features

    for pretrain_seed in PRETRAIN_SEEDS:
        feat = encoder_feature_sets[pretrain_seed]
        for n_labels in LABEL_BUDGETS:
            idx = np.arange(n_labels)

            feature_variants = {
                "physics_adaptive_10": (
                    physics_train[idx],
                    physics_val,
                    {k: v for k, v in physics_split.items()},
                ),
                "scratch_adaptive_10": (
                    feat["scratch"]["train"][idx],
                    feat["scratch"]["validation"],
                    {k: feat["scratch"][k] for k in splits},
                ),
                "krylov_adaptive_10": (
                    feat["krylov"]["train"][idx],
                    feat["krylov"]["validation"],
                    {k: feat["krylov"][k] for k in splits},
                ),
                "projected_krylov_adaptive_10": (
                    feat["projected"]["train"][idx],
                    feat["projected"]["validation"],
                    {k: feat["projected"][k] for k in splits},
                ),
                "krylov_physics_adaptive_10": (
                    np.concatenate(
                        [feat["krylov"]["train"][idx], physics_train[idx]],
                        axis=1,
                    ),
                    np.concatenate(
                        [feat["krylov"]["validation"], physics_val],
                        axis=1,
                    ),
                    {
                        k: np.concatenate(
                            [feat["krylov"][k], physics_split[k]],
                            axis=1,
                        )
                        for k in splits
                    },
                ),
            }

            for method, (x_train, x_val, x_splits) in feature_variants.items():
                model, tuning = choose_ridge(
                    x_train,
                    target_train[idx],
                    x_val,
                    validation,
                    grid,
                )
                tuning_rows.append(
                    {
                        "seed": pretrain_seed,
                        "n_labels": n_labels,
                        "method": method,
                        **tuning,
                    }
                )
                for split_name, samples in splits.items():
                    pred = model.predict(x_splits[split_name])
                    for sample, coeff in zip(samples, pred):
                        result = adaptive_from_coeff(sample, grid, coeff)
                        rows.append(
                            metric_row(
                                method,
                                sample,
                                result,
                                grid,
                                seed=pretrain_seed,
                                n_labels=n_labels,
                                split=split_name,
                            )
                        )

            x_train = np.concatenate(
                [feat["krylov"]["train"][idx], physics_train[idx]],
                axis=1,
            )
            x_val = np.concatenate(
                [feat["krylov"]["validation"], physics_val],
                axis=1,
            )
            residual_model, tuning = choose_ridge(
                x_train,
                target_train[idx] - pt2_train[idx],
                x_val,
                validation,
                grid,
                pt2_val=pt2_val,
            )
            method = "krylov_pt2_residual_adaptive_10"
            tuning_rows.append(
                {
                    "seed": pretrain_seed,
                    "n_labels": n_labels,
                    "method": method,
                    **tuning,
                }
            )
            for split_name, samples in splits.items():
                x_test = np.concatenate(
                    [feat["krylov"][split_name], physics_split[split_name]],
                    axis=1,
                )
                pred_residual = residual_model.predict(x_test)
                pt2_test = pt2_matrix(samples, grid)
                for sample, c_pt2, c_res in zip(samples, pt2_test, pred_residual):
                    result = adaptive_from_coeff(
                        sample,
                        grid,
                        c_pt2 + c_res,
                    )
                    rows.append(
                        metric_row(
                            method,
                            sample,
                            result,
                            grid,
                            seed=pretrain_seed,
                            n_labels=n_labels,
                            split=split_name,
                        )
                    )

                    result11 = adaptive_two_direction(
                        sample,
                        grid,
                        c_pt2,
                        c_res,
                    )
                    rows.append(
                        metric_row(
                            "krylov_pt2_two_direction_11",
                            sample,
                            result11,
                            grid,
                            seed=pretrain_seed,
                            n_labels=n_labels,
                            split=split_name,
                        )
                    )

    aggregate_rows = aggregate(rows)
    write_csv(out / "per_example.csv", rows)
    write_csv(out / "aggregate.csv", aggregate_rows)
    write_csv(out / "tuning.csv", tuning_rows)
    write_csv(out / "encoder_diagnostics.csv", encoder_diag)

    comparisons = {
        "krylov10_vs_fixed25_residual": bootstrap_comparison(
            rows,
            "krylov_pt2_residual_adaptive_10",
            "fixed_ritz_25",
            PRIMARY_LABEL_BUDGET,
            "test_ID",
            "residual_true_e",
        ),
        "krylov10_vs_scratch10_residual": bootstrap_comparison(
            rows,
            "krylov_pt2_residual_adaptive_10",
            "scratch_adaptive_10",
            PRIMARY_LABEL_BUDGET,
            "test_ID",
            "residual_true_e",
        ),
        "krylov10_vs_pt2_10_residual": bootstrap_comparison(
            rows,
            "krylov_pt2_residual_adaptive_10",
            "pt2_adaptive_10",
            PRIMARY_LABEL_BUDGET,
            "test_ID",
            "residual_true_e",
        ),
    }

    def get_agg(method: str, n_labels: int, split: str) -> dict:
        matches = [
            r
            for r in aggregate_rows
            if r["method"] == method
            and int(r["n_labels"]) == n_labels
            and r["split"] == split
        ]
        if not matches:
            raise KeyError((method, n_labels, split))
        return matches[0]

    primary = get_agg(
        "krylov_pt2_residual_adaptive_10",
        PRIMARY_LABEL_BUDGET,
        "test_ID",
    )
    fixed25 = get_agg("fixed_ritz_25", 0, "test_ID")
    scratch10 = get_agg("scratch_adaptive_10", PRIMARY_LABEL_BUDGET, "test_ID")
    pt2 = get_agg("pt2_adaptive_10", 0, "test_ID")

    success = {
        "basis_efficiency_breakthrough": bool(
            primary["basis_dim_mean"] < fixed25["basis_dim_mean"]
            and primary["residual_true_e_mean"] < fixed25["residual_true_e_mean"]
        ),
        "krylov_representation_advantage": bool(
            primary["residual_true_e_mean"]
            < scratch10["residual_true_e_mean"]
        ),
        "learned_correction_beats_pt2": bool(
            primary["residual_true_e_mean"] < pt2["residual_true_e_mean"]
        ),
        "paired_ci_beats_fixed25": bool(
            comparisons["krylov10_vs_fixed25_residual"]["ci_high"] < 0
        ),
        "paired_ci_beats_scratch10": bool(
            comparisons["krylov10_vs_scratch10_residual"]["ci_high"] < 0
        ),
    }

    best_seed = PRETRAIN_SEEDS[0]
    n_labels = PRIMARY_LABEL_BUDGET
    feat = encoder_feature_sets[best_seed]
    x_train = np.concatenate(
        [feat["krylov"]["train"][:n_labels], physics_train[:n_labels]],
        axis=1,
    )
    x_val = np.concatenate(
        [feat["krylov"]["validation"], physics_val],
        axis=1,
    )
    model, _ = choose_ridge(
        x_train,
        target_train[:n_labels] - pt2_train[:n_labels],
        x_val,
        validation,
        grid,
        pt2_val=pt2_val,
    )
    x_test = np.concatenate(
        [feat["krylov"]["test_ID"], physics_split["test_ID"]],
        axis=1,
    )
    c_res = model.predict(x_test)
    c_pt2 = pt2_matrix(test_id, grid)
    hybrid_starts = [
        adaptive_from_coeff(s, grid, a + b).wavefunction.reshape(-1)
        for s, a, b in zip(test_id, c_pt2, c_res)
    ]
    ritz9_starts = [
        fixed_ritz(s, grid, 3).wavefunction.reshape(-1)
        for s in test_id
    ]
    rng = np.random.default_rng(909)
    random_starts = [
        rng.normal(size=grid.n_dof)
        for _ in test_id
    ]
    convergence = {
        "hybrid10": convergence_curve(test_id, grid, hybrid_starts),
        "ritz9": convergence_curve(test_id, grid, ritz9_starts),
        "random": convergence_curve(test_id, grid, random_starts),
    }
    write_json(convergence, out / "krylov_convergence.json")

    summary = {
        "status": "ok",
        "protocol": {
            "grid": 16,
            "low_basis_dim": 9,
            "adaptive_dim": 10,
            "fixed_comparator_dim": 25,
            "full_spectral_side": FULL_SIDE,
            "pretrain_seeds": PRETRAIN_SEEDS,
            "label_budgets": LABEL_BUDGETS,
            "primary_label_budget": PRIMARY_LABEL_BUDGET,
            "fresh_seed_ranges": FRESH_SEEDS,
            "ridge_alphas": RIDGE_ALPHAS,
        },
        "primary": primary,
        "fixed25": fixed25,
        "scratch10": scratch10,
        "pt2_adaptive10": pt2,
        "comparisons": comparisons,
        "success": success,
        "convergence": convergence,
        "elapsed_sec": time.time() - t0,
        "claim_boundary": (
            "This is a fresh finite synthetic benchmark. A breakthrough gate only "
            "means the adaptive 10D solver passed its preregistered comparisons on "
            "this protocol; it does not establish universal superiority."
        ),
    }
    write_json(summary, out / "summary.json")

    lines = [
        "# Hybrid spectral breakthrough study",
        "",
        "The learned component proposes high-frequency directions; all reported final "
        "eigenpairs come from exact Rayleigh--Ritz solves inside the selected subspace.",
        "",
        "## Primary comparison",
        "",
        "| Method | Basis dim | Fidelity | Residual @ true E | Relative energy error |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in [primary, scratch10, pt2, fixed25]:
        lines.append(
            f"| {r['method']} | {r['basis_dim_mean']:.0f} | "
            f"{r['fidelity_mean']:.6f} | {r['residual_true_e_mean']:.6f} | "
            f"{r['rel_energy_error_mean']:.6f} |"
        )
    lines += [
        "",
        "## Gates",
        "",
    ]
    for key, value in success.items():
        lines.append(f"- {key}: **{value}**")
    lines += [
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    (out / "HYBRID_BREAKTHROUGH_REPORT.md").write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
