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

def reconstruct(coeff: np.ndarray, grid: GridSpec, side: int) -> np.ndarray:
    b, _ = sine_basis(grid, side)
    return normalize_wavefunction((b @ np.asarray(coeff)).reshape(grid.ny, grid.nx), grid)


def perturbation_predict(potential: np.ndarray, grid: GridSpec, side: int = 5) -> tuple[float, np.ndarray, np.ndarray]:
    b, _ = sine_basis(grid, side)
    ham0 = build_hamiltonian(grid, np.zeros_like(potential))
    e0 = np.diag(b.T @ (ham0.matrix @ b))
    vmat = b.T @ (potential.ravel()[:, None] * b)
    coeff = np.zeros(b.shape[1]); coeff[0] = 1.0
    for j in range(1, len(coeff)):
        denom = e0[0] - e0[j]
        coeff[j] = 0.0 if abs(denom) < 1e-12 else vmat[j, 0] / denom
    energy = float(e0[0] + vmat[0, 0])
    psi = normalize_wavefunction((b @ coeff).reshape(grid.ny, grid.nx), grid)
    return energy, psi, coeff


def krylov_ritz(potential: np.ndarray, grid: GridSpec, depth: int, seed: int, q0: np.ndarray | None = None) -> tuple[float, np.ndarray, float]:
    ham = build_hamiltonian(grid, potential)
    lr = run_lanczos(ham, q0=q0, depth=max(1, depth), q0_seed=seed)
    q = lr.q[:-1].T
    hp = q.T @ (ham.matrix @ q); hp = (hp + hp.T) / 2
    vals, vecs = np.linalg.eigh(hp)
    p = q @ vecs[:, 0]
    psi = normalize_wavefunction(p.reshape(grid.ny, grid.nx), grid)
    residual = schrodinger_residual(potential, psi, float(vals[0]), grid)
    return float(vals[0]), psi, residual


def block_krylov_ritz(potential: np.ndarray, grid: GridSpec, depth: int, n_starts: int, seed: int) -> tuple[float, np.ndarray, float]:
    ham = build_hamiltonian(grid, potential)
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(ham.n_dof, n_starts))
    q, _ = np.linalg.qr(q)
    blocks = [q]
    cur = q
    for _ in range(depth):
        cur = ham.matrix @ cur
        cur = cur - np.concatenate(blocks, axis=1) @ (np.concatenate(blocks, axis=1).T @ cur)
        cur, _ = np.linalg.qr(cur)
        blocks.append(cur)
    B = np.concatenate(blocks, axis=1)
    B, _ = np.linalg.qr(B)
    hp = B.T @ (ham.matrix @ B); hp = (hp + hp.T) / 2
    vals, vecs = np.linalg.eigh(hp)
    psi = normalize_wavefunction((B @ vecs[:, 0]).reshape(grid.ny, grid.nx), grid)
    return float(vals[0]), psi, schrodinger_residual(potential, psi, float(vals[0]), grid)


def cheb_filter(potential: np.ndarray, grid: GridSpec, seed: int, degree: int = 8, tau: float = 0.08) -> tuple[np.ndarray, float, float]:
    ham = build_hamiltonian(grid, potential)
    emin = float(eigsh(ham.matrix, k=1, which='SA', return_eigenvectors=False)[0])
    emax = float(eigsh(ham.matrix, k=1, which='LA', return_eigenvectors=False)[0])
    rng = np.random.default_rng(seed); q = rng.normal(size=ham.n_dof); q /= np.linalg.norm(q)
    xs = np.linspace(-1, 1, 256)
    es = (xs + 1) * 0.5 * (emax - emin) + emin
    c = np.polynomial.chebyshev.chebfit(xs, np.exp(-tau * (es - emin)), degree)
    def A(v: np.ndarray) -> np.ndarray:
        return (2.0 * ham.matvec(v) - (emax + emin) * v) / max(emax - emin, 1e-12)
    t0 = q.copy(); accum = c[0] * t0
    if degree >= 1:
        t1 = A(q); accum = accum + c[1] * t1
    else: t1 = t0
    for k in range(2, degree + 1):
        t2 = 2 * A(t1) - t0
        accum = accum + c[k] * t2
        t0, t1 = t1, t2
    accum /= max(np.linalg.norm(accum), 1e-15)
    return accum, emin, emax


def principal_angle(U: np.ndarray, V: np.ndarray) -> float:
    u, _ = np.linalg.qr(U); v, _ = np.linalg.qr(V)
    s = np.linalg.svd(u.T @ v, compute_uv=False)
    return float(np.arccos(np.clip(np.min(s), -1.0, 1.0)))


def ipr(psi: np.ndarray, grid: GridSpec) -> float:
    p = psi.ravel(); area = cell_area(grid)
    return float(np.sum(p ** 4) * area)


def node_count(psi: np.ndarray) -> int:
    s = np.sign(psi); s[s == 0] = 1
    return int(np.sum(s[:, 1:] != s[:, :-1]) + np.sum(s[1:, :] != s[:-1, :]))


def error_spectrum(pred: np.ndarray, true: np.ndarray, grid: GridSpec, side: int = 6) -> tuple[np.ndarray, np.ndarray]:
    b, labels = sine_basis(grid, min(side, grid.n_interior))
    e = pred.ravel() - (true if np.dot(pred.ravel(), true.ravel()) >= 0 else -true).ravel()
    c = b.T @ e
    freq = np.sum(labels ** 2, axis=1).astype(float)
    return freq, c ** 2


def h1_error(pred: np.ndarray, true: np.ndarray, grid: GridSpec) -> float:
    p = pred if np.dot(pred.ravel(), true.ravel()) >= 0 else -pred
    e = p - true
    gy, gx = np.gradient(e, grid.hy, grid.hx)
    return float(np.sqrt(np.mean(e * e) + np.mean(gx * gx + gy * gy)))


def energy_variance(potential: np.ndarray, psi: np.ndarray, grid: GridSpec) -> float:
    ham = build_hamiltonian(grid, potential); q = psi.ravel(); q = q / np.linalg.norm(q)
    hq = ham.matvec(q); e = float(q @ hq)
    return float(hq @ hq - e * e)


def roughness(v: np.ndarray) -> float:
    gy, gx = np.gradient(v); return float(np.mean(gx * gx + gy * gy))


def custom_potential(kind: str, grid: GridSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); X, Y = grid.meshgrid()
    if kind == 'smooth':
        v = rng.normal(size=(grid.ny, grid.nx)); return 4.0 * gaussian_filter(v, sigma=max(1, grid.nx/6))
    if kind == 'periodic':
        return -4*np.cos(2*np.pi*X)*np.cos(2*np.pi*Y) + 1.5*np.cos(4*np.pi*X)
    if kind == 'disorder':
        return gaussian_filter(rng.uniform(-8, 3, size=(grid.ny, grid.nx)), sigma=0.45)
    if kind == 'barrier_well':
        well = -10*np.exp(-((X-.3)**2+(Y-.5)**2)/(2*.08**2)); barrier = 12*np.exp(-((X-.65)**2)/(2*.035**2)); return well+barrier
    if kind == 'harmonic':
        return 35*((X-.45)**2 + 1.7*(Y-.55)**2)
    if kind == 'quartic':
        return 90*((X-.5)**4 + .6*(Y-.5)**4) + 5*(X-.5)*(Y-.5)
    if kind == 'multiscale':
        return -7*np.exp(-((X-.5)**2+(Y-.5)**2)/(2*.23**2)) - 10*np.exp(-((X-.28)**2+(Y-.7)**2)/(2*.035**2))
    if kind == 'hard_double':
        return -18*np.exp(-((X-.27)**2+(Y-.5)**2)/(2*.07**2)) - 18*np.exp(-((X-.73)**2+(Y-.5)**2)/(2*.07**2)) + 5*np.exp(-((X-.5)**2)/(2*.04**2))
    if kind == 'composite':
        return custom_potential('periodic', grid, seed) + .5*custom_potential('multiscale', grid, seed+1)
    raise ValueError(kind)


def laplacian_dense(n: int, bc: str) -> np.ndarray:
    h = 1.0/(n+1 if bc == 'dirichlet' else n)
    L = np.zeros((n*n, n*n))
    def idx(y,x): return y*n+x
    for y in range(n):
        for x in range(n):
            i=idx(y,x); L[i,i] = -4/h**2
            for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                yy,xx=y+dy,x+dx
                if 0 <= yy < n and 0 <= xx < n:
                    L[i,idx(yy,xx)] += 1/h**2
                elif bc == 'periodic':
                    L[i,idx(yy % n, xx % n)] += 1/h**2
                elif bc == 'neumann':
                    L[i,i] += 1/h**2
    return L


def magnetic_hamiltonian(n: int, v: np.ndarray, ax: float=.7, ay: float=-.4) -> np.ndarray:
    h=1.0/(n+1); t=.5/h**2; H=np.zeros((n*n,n*n),dtype=np.complex128)
    def idx(y,x): return y*n+x
    for y in range(n):
        for x in range(n):
            i=idx(y,x); H[i,i]=4*t+v[y,x]
            for dy,dx,phase in [(0,1,np.exp(1j*ax*h)),(0,-1,np.exp(-1j*ax*h)),(1,0,np.exp(1j*ay*h)),(-1,0,np.exp(-1j*ay*h))]:
                yy,xx=y+dy,x+dx
                if 0<=yy<n and 0<=xx<n: H[i,idx(yy,xx)] += -t*phase
    return H


def linear_ridge(X: np.ndarray, Y: np.ndarray, alpha: float=1e-3) -> tuple[np.ndarray, np.ndarray]:
    xm=X.mean(0); ym=Y.mean(0); xc=X-xm; yc=Y-ym
    if X.shape[0] <= X.shape[1]:
        W=xc.T @ np.linalg.solve(xc@xc.T + alpha*np.eye(len(X)), yc)
    else:
        W=np.linalg.solve(xc.T@xc + alpha*np.eye(X.shape[1]), xc.T@yc)
    return W, ym - xm@W


def ridge_predict(model: tuple[np.ndarray,np.ndarray], X: np.ndarray) -> np.ndarray:
    W,b=model; return X@W+b


class SpectralMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int=64, latent: int=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n_in,hidden),nn.GELU(),nn.Linear(hidden,latent),nn.GELU())
        self.head=nn.Linear(latent,n_out)
    def forward(self,x): return self.head(self.encoder(x))


class MomentNet(nn.Module):
    def __init__(self,n_in:int,n_mom:int,hidden:int=64,latent:int=24):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(n_in,hidden),nn.GELU(),nn.Linear(hidden,latent),nn.LayerNorm(latent),nn.GELU()); self.head=nn.Linear(latent,n_mom)
    def forward(self,x): return self.head(self.encoder(x))


class TinyGraphEnergy(nn.Module):
    def __init__(self, n: int, hidden: int=16):
        super().__init__(); self.n=n; self.lin1=nn.Linear(2,hidden); self.lin2=nn.Linear(hidden,hidden); self.head=nn.Linear(hidden,1)
    def forward(self,v):
        x=v.unsqueeze(-1); p=torch.nn.functional.pad(v.unsqueeze(1),(1,1,1,1)); neigh=(p[:,:,1:-1,2:]+p[:,:,1:-1,:-2]+p[:,:,2:,1:-1]+p[:,:,:-2,1:-1])/4
        h=torch.cat([x,neigh.squeeze(1).unsqueeze(-1)],-1); h=torch.relu(self.lin1(h)); h=torch.relu(self.lin2(h)); return self.head(h.mean((1,2))).squeeze(-1)


class TinyOperatorTransformer(nn.Module):
    def __init__(self,n:int,d:int=24,heads:int=4):
        super().__init__(); self.n=n; self.inp=nn.Linear(3,d); layer=nn.TransformerEncoderLayer(d,heads,2*d,batch_first=True); self.enc=nn.TransformerEncoder(layer,2); self.head=nn.Linear(d,1)
        coords=np.stack(np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='xy'),-1).reshape(-1,2); self.register_buffer('coords',torch.tensor(coords,dtype=torch.float32))
    def forward(self,v):
        b=v.shape[0]; tok=torch.cat([v.reshape(b,-1,1),self.coords.unsqueeze(0).expand(b,-1,-1)],-1); h=self.enc(self.inp(tok)); return self.head(h.mean(1)).squeeze(-1)


def train_regressor(model: nn.Module, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float=2e-3, weight_decay:float=1e-4, anchor:dict[str,torch.Tensor]|None=None, anchor_weight:float=0.0) -> list[float]:
    torch.manual_seed(7); model.train(); xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(Y,dtype=torch.float32); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); hist=[]
    for _ in range(max(1,epochs)):
        pred=model(xt); loss=torch.mean((pred-yt)**2)
        if anchor is not None and anchor_weight>0:
            reg=torch.zeros((),dtype=loss.dtype)
            for name,p in model.named_parameters(): reg=reg+torch.mean((p-anchor[name])**2)
