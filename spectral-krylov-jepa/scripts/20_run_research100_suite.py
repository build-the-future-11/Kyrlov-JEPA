#!/usr/bin/env python3
"""Executable Research-100 suite for Spectral Krylov-JEPA.

Every idea from the 100-item research roadmap has an executable handler. Some
handlers are complete analyses/benchmarks while ambitious architecture ideas are
bounded prototypes. The maturity field is part of every result so prototypes
cannot be mistaken for confirmatory efficacy evidence.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import sparse
from scipy.ndimage import gaussian_filter, zoom
from scipy.sparse.linalg import eigsh, spsolve
from scipy.stats import spearmanr
from torch import nn

from spectral_krylov_jepa.evaluation.metrics import (
    evaluate_example,
    fidelity_np,
    rayleigh_quotient,
    schrodinger_residual,
)
from spectral_krylov_jepa.physics.eigensolver import normalize_wavefunction
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.physics.lanczos import run_lanczos
from spectral_krylov_jepa.physics.potentials import generate_potential

IDEAS: list[tuple[int, str]] = [
(1, 'Turn Krylov-JEPA into a learned eigensolver accelerator'),
(2, 'Neural Rayleigh-Ritz'),
(3, 'Predict a correction to Ritz rather than the full state'),
(4, 'Residual-corrected hybrid'),
(5, 'One-step neural → exact correction'),
(6, 'Chebyshev-filtered Krylov pretraining'),
(7, 'Imaginary-time targets'),
(8, 'Actual Krylov-Ritz target instead of fixed sine Ritz'),
(9, 'Subspace prediction instead of vector prediction'),
(10, 'Principal-angle loss'),
(11, 'Block Krylov-JEPA'),
(12, 'q0-invariance objective'),
(13, 'Same-potential / different-start contrastive positives'),
(14, 'Multi-start consensus teacher'),
(15, 'Learn Lanczos coefficients'),
(16, 'Predict spectral moments'),
(17, 'Stochastic Lanczos quadrature targets'),
(18, 'Predict the eigengap'),
(19, 'Eigengap-conditioned losses'),
(20, 'Predict multiple eigenstates'),
(21, 'Gram-Schmidt layer for learned eigenfunctions'),
(22, 'Remove the learned energy head entirely'),
(23, 'Pure variational fine-tuning'),
(24, 'Energy-variance loss'),
(25, 'Sobolev H1 or energy-norm supervision'),
(26, 'Frequency-weighted error loss'),
(27, 'Ground-state positivity constraint'),
(28, 'No-node prior'),
(29, 'Symmetry-equivariant augmentation'),
(30, 'Predict perturbative coefficients'),
(31, 'Perturbation-theory baseline'),
(32, 'Higher-order perturbation network'),
(33, 'Adaptive basis size'),
(34, 'Learned basis dictionary'),
(35, 'Low-rank Hamiltonian embedding'),
(36, 'Graph representation of the Hamiltonian'),
(37, 'Sparse-matrix transformer'),
(38, 'Neural operator encoder'),
(39, '16→32→64 zero-shot resolution study'),
(40, 'Mesh-independent spectral decoder'),
(41, 'Irregular domains'),
(42, 'Different boundary conditions'),
(43, 'Parameter-conditioned Hamiltonians'),
(44, 'Magnetic/vector-potential extension'),
(45, 'Random smooth fields instead of Gaussian wells'),
(46, 'Periodic/lattice potentials'),
(47, 'Disordered potentials / Anderson-like landscapes'),
(48, 'Barriers + wells'),
(49, 'Anisotropic harmonic traps'),
(50, 'Quartic and anharmonic potentials'),
(51, 'Multi-scale potentials'),
(52, 'Hard double wells with tunneling'),
(53, 'Adversarial potential generator'),
(54, 'Difficulty defined by baseline error'),
(55, 'Difficulty curves instead of one benchmark'),
(56, 'Measure Ritz dimension to tolerance'),
(57, 'Localization score / inverse participation ratio'),
(58, 'Eigengap stratification'),
(59, 'Spectral-frequency error analysis'),
(60, 'Error attribution by spectral band'),
(61, 'Davis-Kahan style analysis'),
(62, 'Formal fidelity-residual bounds'),
(63, 'Upper bound with spectral radius'),
(64, 'Explain ridge theoretically using perturbation theory'),
(65, 'CKA/SVCCA between representations'),
(66, 'Linear probes for physical quantities'),
(67, 'Probe after every pretraining checkpoint'),
(68, 'Transfer-geometry plot'),
(69, 'Freeze-vs-unfreeze study'),
(70, 'Linear-probe-only downstream'),
(71, 'Gradual unfreezing'),
(72, 'L2-SP / anchoring to pretrained weights'),
(73, 'Separate encoder learning rates by layer'),
(74, 'Adapter fine-tuning'),
(75, 'LoRA on the encoder'),
(76, 'Meta-learning across potential families'),
(77, 'Leave-one-family-out transfer'),
(78, 'Cross-resolution adaptation'),
(79, 'Cross-boundary-condition transfer'),
(80, 'Cross-operator transfer'),
(81, 'Uncertainty estimates'),
(82, 'Failure detector'),
(83, 'Selective hybrid solver'),
(84, 'Adaptive compute'),
(85, 'Learn stopping criterion'),
(86, 'Learn optimal shift for shift-invert'),
(87, 'Learn a diagonal preconditioner'),
(88, 'Learn multigrid-style correction'),
(89, 'Differentiable Lanczos unrolling'),
(90, 'Loss = iteration count proxy'),
(91, 'Principal-angle convergence loss'),
(92, 'Amortized preconditioner + Ritz hybrid'),
(93, 'Measure real computational economics'),
(94, 'Cold-start vs amortized setting'),
(95, 'Learning curves in unlabeled compute'),
(96, 'Scaling law'),
(97, 'Scaling in model size'),
(98, 'Scaling in Krylov depth'),
(99, 'Scaling in number of starts'),
(100, 'When Does Krylov Pretraining Help phase diagram'),
]

CATEGORY = {}
for i, _ in IDEAS:
    if i <= 5: CATEGORY[i] = 'solver_alignment'
    elif i <= 19: CATEGORY[i] = 'spectral_pretraining'
    elif i <= 29: CATEGORY[i] = 'physics_objectives'
    elif i <= 44: CATEGORY[i] = 'architectures_operators'
    elif i <= 64: CATEGORY[i] = 'benchmark_theory'
    elif i <= 80: CATEGORY[i] = 'representation_transfer'
    elif i <= 94: CATEGORY[i] = 'hybrid_solver_systems'
    else: CATEGORY[i] = 'scaling_phase_diagram'

MATURITY = {i: 'prototype' for i, _ in IDEAS}
for i in list(range(45, 65)) + [93, 94, 95, 96, 98, 99, 100]:
    MATURITY[i] = 'benchmark'
for i in [24, 25, 26, 27, 28, 29, 41, 42, 43, 44, 57, 58, 59, 60, 61, 62, 63, 65, 66, 68, 81]:
    MATURITY[i] = 'analysis'
for i in [1, 2, 3, 4, 5, 23, 30, 32, 33, 34, 36, 37, 38, 69, 70, 71, 72, 73, 74, 75, 76, 77, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97]:
    MATURITY[i] = 'trainable'


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    return x


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float('nan')


def sine_basis(grid: GridSpec, side: int) -> tuple[np.ndarray, np.ndarray]:
    x = (grid.x_coords() - grid.x_min) / (grid.x_max - grid.x_min)
    y = (grid.y_coords() - grid.y_min) / (grid.y_max - grid.y_min)
    cols, labels = [], []
    for my in range(1, side + 1):
        for mx in range(1, side + 1):
            q = np.outer(np.sin(np.pi * my * y), np.sin(np.pi * mx * x)).ravel()
            q /= max(np.linalg.norm(q), 1e-15)
            cols.append(q); labels.append((mx, my))
    return np.stack(cols, axis=1), np.asarray(labels, dtype=int)


def ritz(potential: np.ndarray, grid: GridSpec, side: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    basis, labels = sine_basis(grid, side)
    ham = build_hamiltonian(grid, potential)
    hb = ham.matrix @ basis
    hp = (basis.T @ hb)
    hp = (hp + hp.T) / 2.0
    vals, vecs = np.linalg.eigh(hp)
    coeff = vecs[:, 0]
    psi = normalize_wavefunction((basis @ coeff).reshape(grid.ny, grid.nx), grid)
    return float(vals[0]), psi, coeff, vals


def solve_k(potential: np.ndarray, grid: GridSpec, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    ham = build_hamiltonian(grid, potential)
    k_use = min(k, ham.n_dof - 2)
    vals, vecs = eigsh(ham.matrix, k=k_use, which='SA', tol=1e-9, maxiter=5000)
    order = np.argsort(vals); vals = vals[order]; vecs = vecs[:, order]
    out = []
    for j in range(k_use):
        psi = normalize_wavefunction(vecs[:, j].reshape(grid.ny, grid.nx), grid)
        flat = psi.ravel(); idx = int(np.argmax(np.abs(flat)))
        if flat[idx] < 0: psi = -psi
        out.append(psi)
    return vals.astype(float), np.stack(out, axis=0)


def spectral_coeff(psi: np.ndarray, grid: GridSpec, side: int) -> np.ndarray:
    b, _ = sine_basis(grid, side)
    # Euclidean-normalized basis/state coefficients; global scale is irrelevant before reconstruction.
    p = psi.ravel().astype(float); p /= max(np.linalg.norm(p), 1e-15)
    return b.T @ p
