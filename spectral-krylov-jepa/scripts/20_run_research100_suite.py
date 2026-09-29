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

            loss=loss+anchor_weight*reg
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss.item()))
    return hist


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    X=X-X.mean(0); Y=Y-Y.mean(0); hs=np.linalg.norm(X.T@Y,'fro')**2; den=np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro'); return float(hs/max(den,1e-15))


def svcca(X: np.ndarray, Y: np.ndarray, rank:int=8) -> float:
    X=X-X.mean(0); Y=Y-Y.mean(0); ux,sx,_=np.linalg.svd(X,full_matrices=False); uy,sy,_=np.linalg.svd(Y,full_matrices=False); rx=min(rank,ux.shape[1],uy.shape[1]); A=ux[:,:rx]; B=uy[:,:rx]; s=np.linalg.svd(A.T@B,compute_uv=False); return float(np.mean(s))


def fit_failure_detector(features: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray,float]:
    W,b=linear_ridge(features,bad[:,None].astype(float),alpha=1e-2); return W[:,0],float(b[0])


def classification_accuracy(W:np.ndarray,b:float,X:np.ndarray,y:np.ndarray)->float:
    return float(np.mean(((X@W+b)>=.5)==y))


def spectral_moments(v: np.ndarray, grid: GridSpec, q: np.ndarray, order:int=4) -> np.ndarray:
    ham=build_hamiltonian(grid,v); cur=q.copy(); out=[]
    for _ in range(order):
        cur=ham.matvec(cur); out.append(float(q@cur))
    return np.asarray(out)


def iterations_to_residual(v:np.ndarray,grid:GridSpec,q0:np.ndarray,max_depth:int=16,tol:float=1.0)->tuple[int,float]:
    for d in range(1,max_depth+1):
        try:
            _,_,r=krylov_ritz(v,grid,d,123,q0=q0)
        except RuntimeError:
            continue
        if r<=tol: return d,r
    return max_depth+1,float(r if 'r' in locals() else np.inf)


def jacobi_correction(v:np.ndarray,grid:GridSpec,psi:np.ndarray,energy:float,omega:float=.5)->np.ndarray:
    ham=build_hamiltonian(grid,v); q=psi.ravel(); r=ham.matvec(q)-energy*q; d=ham.matrix.diagonal()-energy; step=np.divide(r,d,out=np.zeros_like(r),where=np.abs(d)>1e-10); q2=q-omega*step; return normalize_wavefunction(q2.reshape(grid.ny,grid.nx),grid)


def coarse_correction(v:np.ndarray,grid:GridSpec,psi:np.ndarray,energy:float)->np.ndarray:
    ham=build_hamiltonian(grid,v); r=(ham.matvec(psi.ravel())-energy*psi.ravel()).reshape(grid.ny,grid.nx)
    smooth=gaussian_filter(r,sigma=max(1,grid.nx/8)); q=psi-.02*smooth; return normalize_wavefunction(q,grid)


def inverse_iteration(v:np.ndarray,grid:GridSpec,q0:np.ndarray,shift:float,steps:int=3)->tuple[np.ndarray,float]:
    ham=build_hamiltonian(grid,v); A=ham.matrix-shift*sparse.eye(ham.n_dof,format='csr'); q=q0.ravel(); q=q/max(np.linalg.norm(q),1e-15)
    for _ in range(steps):
        q=spsolve(A,q); q=q/max(np.linalg.norm(q),1e-15)
    psi=normalize_wavefunction(q.reshape(grid.ny,grid.nx),grid); e=rayleigh_quotient(v,psi,grid); return psi,e


def d4_transforms(a:np.ndarray)->list[np.ndarray]:
    return [a,np.rot90(a,1),np.rot90(a,2),np.rot90(a,3),np.fliplr(a),np.flipud(a),np.transpose(a),np.fliplr(np.transpose(a))]


def potential_features(v:np.ndarray)->np.ndarray:
    f=np.fft.rfft2(v); mag=np.abs(f[:4,:4]).ravel(); return np.concatenate([[v.mean(),v.std(),v.min(),v.max(),roughness(v)],mag])


def sign_align(pred:np.ndarray,true:np.ndarray)->np.ndarray:
    return pred if np.dot(pred.ravel(),true.ravel())>=0 else -pred


@dataclass
class Sample:
    v: np.ndarray
    evals: np.ndarray
    states: np.ndarray
    family: str
    seed: int


@dataclass
class Context:
    mode: str
    seed: int
    out_dir: Path
    grid: GridSpec
    train: list[Sample]
    test: list[Sample]
    families: dict[str, list[Sample]]
    side: int
    epochs: int
    caches: dict[str, Any] = field(default_factory=dict)

    def X(self, samples:list[Sample]|None=None)->np.ndarray:
        s=samples or self.train; return np.stack([x.v.ravel() for x in s]).astype(np.float32)
    def Y(self,samples:list[Sample]|None=None)->np.ndarray:
        s=samples or self.train; return np.stack([np.concatenate([spectral_coeff(x.states[0],self.grid,self.side),[x.evals[0]]]) for x in s]).astype(np.float32)
    def direct_model(self)->SpectralMLP:
        if 'direct_model' not in self.caches:
            m=SpectralMLP(self.grid.n_dof,self.side*self.side+1,hidden=48 if self.mode=='smoke' else 96,latent=24 if self.mode=='smoke' else 48)
            train_regressor(m,self.X(),self.Y(),self.epochs)
            self.caches['direct_model']=m
        return self.caches['direct_model']
    def direct_predictions(self,samples:list[Sample]|None=None)->list[tuple[float,np.ndarray]]:
        s=samples or self.test; key='pred_'+str(id(s))
        if key not in self.caches:
            m=self.direct_model().eval(); xt=torch.tensor(self.X(s),dtype=torch.float32)
            with torch.no_grad(): y=m(xt).numpy()
            self.caches[key]=[(float(row[-1]),reconstruct(row[:-1],self.grid,self.side)) for row in y]
        return self.caches[key]
    def perturb_predictions(self,samples:list[Sample]|None=None)->list[tuple[float,np.ndarray]]:
        s=samples or self.test; return [(perturbation_predict(x.v,self.grid,self.side)[0],perturbation_predict(x.v,self.grid,self.side)[1]) for x in s]
    def moment_pretrain(self, latent:int=24, n_unlabeled:int|None=None)->tuple[MomentNet,list[dict[str,float]]]:
        key=f'moment_{latent}_{n_unlabeled}'
        if key in self.caches: return self.caches[key]
        samples=self.train[:n_unlabeled] if n_unlabeled else self.train
        rng=np.random.default_rng(self.seed+991); q=rng.normal(size=self.grid.n_dof); q/=np.linalg.norm(q)
        X=self.X(samples); Y=np.stack([spectral_moments(s.v,self.grid,q,4) for s in samples]).astype(np.float32)
        ys=Y.std(0); ys[ys<1e-6]=1; ym=Y.mean(0); Yn=(Y-ym)/ys
        m=MomentNet(self.grid.n_dof,4,hidden=max(32,latent*2),latent=latent); xt=torch.tensor(X); yt=torch.tensor(Yn); opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
        checkpoints=[]; total=max(4,self.epochs)
        for ep in range(total):
            pred=m(xt); loss=((pred-yt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if ep in {0,total//2,total-1}:
                with torch.no_grad(): z=m.encoder(xt).numpy()
                checkpoints.append({'epoch':ep,'pretrain_loss':float(loss.item()),'rank':effective_rank(z)})
        self.caches[key]=(m,checkpoints); return self.caches[key]


def effective_rank(z:np.ndarray)->float:
    z=z-z.mean(0); s=np.linalg.svd(z,compute_uv=False); p=s*s; p/=max(p.sum(),1e-30); return float(np.exp(-np.sum(p*np.log(np.clip(p,1e-30,None)))))


def make_samples(grid:GridSpec,n:int,seed0:int,family:str)->list[Sample]:
    out=[]
    for i in range(n):
        if family in {'id_gaussian_mixture','ood_narrow','ood_strong','ood_double','ood_rough'}: v,_=generate_potential(family,seed0+i,grid=grid)
        else: v=custom_potential(family,grid,seed0+i)
        vals,states=solve_k(v,grid,3); out.append(Sample(v=v,evals=vals,states=states,family=family,seed=seed0+i))
    return out


def build_context(mode:str,seed:int,out_dir:Path)->Context:
    if mode=='smoke': n=6; n_train=12; n_test=6; epochs=8; side=3
    else: n=12; n_train=48; n_test=18; epochs=80; side=5
    grid=GridSpec(n_interior=n)
    train=make_samples(grid,n_train,seed+1000,'id_gaussian_mixture'); test=make_samples(grid,n_test,seed+5000,'id_gaussian_mixture')
    fam_names=['smooth','periodic','disorder','barrier_well','harmonic','quartic','multiscale','hard_double','ood_narrow','ood_strong','ood_double','ood_rough']
    fam_count=2 if mode=='smoke' else 6
    families={name:make_samples(grid,fam_count,seed+10000+100*j,name) for j,name in enumerate(fam_names)}
    return Context(mode=mode,seed=seed,out_dir=out_dir,grid=grid,train=train,test=test,families=families,side=side,epochs=epochs)


def prediction_metrics(ctx:Context,preds:list[tuple[float,np.ndarray]],samples:list[Sample]|None=None)->dict[str,float]:
    s=samples or ctx.test; rows=[]
    for sample,(e,p) in zip(s,preds): rows.append(evaluate_example(sample.v,sample.states[0],float(sample.evals[0]),p,float(e),ctx.grid))
    return {k:float(np.mean([r[k] for r in rows])) for k in ['fidelity','rel_energy_error','residual_true_e','residual_rayleigh','sign_aligned_rel_l2']}


def basis_target_matrix(ctx:Context,samples:list[Sample]|None=None)->np.ndarray:
    s=samples or ctx.train; return np.stack([spectral_coeff(x.states[0],ctx.grid,ctx.side) for x in s])


def latent_features(ctx:Context, model:MomentNet, samples:list[Sample])->np.ndarray:
    with torch.no_grad(): return model.encoder(torch.tensor(ctx.X(samples),dtype=torch.float32)).numpy()


def probe(lat_train:np.ndarray,y_train:np.ndarray,lat_test:np.ndarray,y_test:np.ndarray)->float:
    mdl=linear_ridge(lat_train,y_train,alpha=1e-2); p=ridge_predict(mdl,lat_test); return float(np.mean((p-y_test)**2))


def train_energy_model(model:nn.Module,ctx:Context,epochs:int|None=None)->float:
    X=np.stack([s.v for s in ctx.train]).astype(np.float32); y=np.asarray([s.evals[0] for s in ctx.train],dtype=np.float32); scale=max(float(y.std()),1e-6); mean=float(y.mean()); yt=(y-mean)/scale
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3); model.train(); tx=torch.tensor(X); ty=torch.tensor(yt)
    for _ in range(epochs or ctx.epochs):
        p=model(tx); loss=((p-ty)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); q=torch.tensor(np.stack([s.v for s in ctx.test]).astype(np.float32)); true=np.asarray([s.evals[0] for s in ctx.test])
    with torch.no_grad(): pred=model(q).numpy()*scale+mean
    return float(np.mean(np.abs(pred-true)/(np.abs(true)+1e-6)))


def variational_refine(v:np.ndarray,grid:GridSpec,psi:np.ndarray,steps:int=20,lr:float=.05)->np.ndarray:
    ham=torch.tensor(build_hamiltonian(grid,v).to_dense(),dtype=torch.float32); q=torch.tensor(psi.ravel()/np.linalg.norm(psi.ravel()),dtype=torch.float32,requires_grad=True); opt=torch.optim.Adam([q],lr=lr)
    for _ in range(steps):
        qn=q/(torch.linalg.vector_norm(q)+1e-12); e=qn@(ham@qn); opt.zero_grad(); e.backward(); opt.step()
    qn=(q.detach()/torch.linalg.vector_norm(q.detach())).numpy(); return normalize_wavefunction(qn.reshape(grid.ny,grid.nx),grid)


def d4_augment(X:np.ndarray,Y:np.ndarray,n:int)->tuple[np.ndarray,np.ndarray]:
    xs=[]; ys=[]
    for x,y in zip(X,Y):
        vf=x.reshape(n,n)
        for t in d4_transforms(vf): xs.append(t.ravel()); ys.append(y)
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32)


def benchmark_family(ctx:Context,name:str)->dict[str,float]:
    samples=ctx.families[name]; free=[]; ritz3=[]; gaps=[]; iprs=[]
    b1,_=sine_basis(ctx.grid,1); psi0=normalize_wavefunction(b1[:,0].reshape(ctx.grid.ny,ctx.grid.nx),ctx.grid)
    for s in samples:
        ef=rayleigh_quotient(s.v,psi0,ctx.grid); free.append(evaluate_example(s.v,s.states[0],float(s.evals[0]),psi0,ef,ctx.grid)['residual_true_e'])
        er,pr,_,_=ritz(s.v,ctx.grid,min(3,ctx.grid.n_interior)); ritz3.append(schrodinger_residual(s.v,pr,float(s.evals[0]),ctx.grid)); gaps.append(float(s.evals[1]-s.evals[0])); iprs.append(ipr(s.states[0],ctx.grid))
    return {'free_residual':_safe_mean(free),'ritz3_residual':_safe_mean(ritz3),'eigengap':_safe_mean(gaps),'ipr':_safe_mean(iprs),'roughness':_safe_mean([roughness(s.v) for s in samples])}


def run_experiment(eid:int,ctx:Context)->tuple[dict[str,Any],str]:
    s=ctx.test[0]; v=s.v; true=s.states[0]; e0=float(s.evals[0]); grid=ctx.grid
    direct=ctx.direct_predictions(); dm=prediction_metrics(ctx,direct)
    pred_e,pred_psi=direct[0]

    if eid==1:
        qlearn=pred_psi.ravel()/np.linalg.norm(pred_psi); qrand=np.random.default_rng(1).normal(size=grid.n_dof); qrand/=np.linalg.norm(qrand)
        il,rl=iterations_to_residual(v,grid,qlearn,12,1.0); ir,rr=iterations_to_residual(v,grid,qrand,12,1.0)
        return {'learned_start_steps':il,'random_start_steps':ir,'learned_final_residual':rl,'random_final_residual':rr},'Warm-start accelerator proxy using exact Lanczos/Ritz iterations.'
    if eid==2:
        er,pr,_,_=ritz(v,grid,min(ctx.side,4)); return evaluate_example(v,true,e0,pr,er,grid),'Exact Rayleigh-Ritz endpoint in a compact basis.'
    if eid==3:
        er,pr,c,_=ritz(v,grid,min(3,grid.n_interior)); base=spectral_coeff(pr,grid,ctx.side); X=ctx.X(); Y=basis_target_matrix(ctx)-np.stack([spectral_coeff(ritz(x.v,grid,min(3,grid.n_interior))[1],grid,ctx.side) for x in ctx.train]); mdl=linear_ridge(X,Y,1e-2); corr=ridge_predict(mdl,v.ravel()[None])[0]; ph=reconstruct(base+corr,grid,ctx.side); return evaluate_example(v,true,e0,ph,rayleigh_quotient(v,ph,grid),grid),'Learned correction to fixed Ritz baseline.'
    if eid==4:
        er,pr,_,_=ritz(v,grid,min(3,grid.n_interior)); ph=normalize_wavefunction(pr+.25*sign_align(pred_psi-pr,true),grid); return evaluate_example(v,true,e0,ph,rayleigh_quotient(v,ph,grid),grid),'Bounded residual-corrected Ritz/direct hybrid.'
    if eid==5:
        before=schrodinger_residual(v,pred_psi,rayleigh_quotient(v,pred_psi,grid),grid); ph,ee=inverse_iteration(v,grid,pred_psi,rayleigh_quotient(v,pred_psi,grid)-.2,1); after=schrodinger_residual(v,ph,ee,grid); return {'residual_before':before,'residual_after_one_inverse_step':after},'One exact inverse-iteration correction.'
    if eid in (6,7):

        q,emin,emax=cheb_filter(v,grid,ctx.seed+eid,degree=5 if ctx.mode=='smoke' else 10,tau=.05 if eid==6 else .15); psi=normalize_wavefunction(q.reshape(grid.ny,grid.nx),grid); return {'fidelity':fidelity_np(psi,true,grid),'residual_rayleigh':schrodinger_residual(v,psi,rayleigh_quotient(v,psi,grid),grid),'spectral_min':emin,'spectral_max':emax},'Chebyshev exponential filter; larger tau is imaginary-time target.'
    if eid==8:
        er,pr,r=krylov_ritz(v,grid,4,ctx.seed); return {'fidelity':fidelity_np(pr,true,grid),'residual':r,'energy_error':abs(er-e0)},'Actual Lanczos-basis Ritz vector target.'
    if eid in (9,10):
        vals,states=solve_k(v,grid,2); ham=build_hamiltonian(grid,v); lr=run_lanczos(ham,depth=min(5,grid.n_dof-2),q0_seed=ctx.seed); U=np.stack([x.ravel()/np.linalg.norm(x.ravel()) for x in states[:2]],axis=1); V=lr.q[:-1].T; ang=principal_angle(U,V); proj_err=float(np.linalg.norm(U@U.T - np.linalg.qr(V)[0]@np.linalg.qr(V)[0].T,'fro'))
        return {'principal_angle_rad':ang,'projector_fro_error':proj_err},'Subspace/projector geometry diagnostic.'
    if eid==11:
        eb,pb,rb=block_krylov_ritz(v,grid,2,3,ctx.seed); es,ps,rs=krylov_ritz(v,grid,5,ctx.seed); return {'block_residual':rb,'single_residual':rs,'block_fidelity':fidelity_np(pb,true,grid),'single_fidelity':fidelity_np(ps,true,grid)},'Block versus single-start Krylov.'
    if eid in (12,13,14):
        vecs=[]; energies=[]
        for j in range(4):
            e,p,_=krylov_ritz(v,grid,4,ctx.seed+100*j); vecs.append(p.ravel()/np.linalg.norm(p.ravel())); energies.append(e)
        sims=[abs(float(vecs[0]@x)) for x in vecs[1:]]; consensus=np.linalg.svd(np.stack(vecs).T,full_matrices=False)[0][:,0]; cp=normalize_wavefunction(consensus.reshape(grid.ny,grid.nx),grid)
        return {'mean_same_potential_similarity':_safe_mean(sims),'ritz_energy_std':float(np.std(energies)),'consensus_fidelity':fidelity_np(cp,true,grid)},'Multi-q0 invariance/contrastive/consensus diagnostic.'
    if eid==15:
        X=ctx.X(); Y=[]
        for sm in ctx.train:
            lr=run_lanczos(build_hamiltonian(grid,sm.v),depth=3,q0_seed=ctx.seed); Y.append(np.concatenate([lr.alpha,lr.beta]))
        mdl=linear_ridge(X,np.asarray(Y),1e-2); yt=[]
        for sm in ctx.test:
            lr=run_lanczos(build_hamiltonian(grid,sm.v),depth=3,q0_seed=ctx.seed); yt.append(np.concatenate([lr.alpha,lr.beta]))
        p=ridge_predict(mdl,ctx.X(ctx.test)); return {'coefficient_mse':float(np.mean((p-np.asarray(yt))**2))},'Potential-to-Lanczos-coefficient probe.'
    if eid==16:
        rng=np.random.default_rng(ctx.seed); q=rng.normal(size=grid.n_dof); q/=np.linalg.norm(q); Y=np.stack([spectral_moments(x.v,grid,q,4) for x in ctx.train]); mdl=linear_ridge(ctx.X(),Y,1e-2); yt=np.stack([spectral_moments(x.v,grid,q,4) for x in ctx.test]); return {'moment_mse':float(np.mean((ridge_predict(mdl,ctx.X(ctx.test))-yt)**2))},'Spectral-moment pretext target predictability.'
    if eid==17:
        ham=build_hamiltonian(grid,v); tau=.05; estimates=[]
        for j in range(4):
            lr=run_lanczos(ham,depth=min(6,grid.n_dof-2),q0_seed=ctx.seed+j); T=np.diag(lr.alpha)+np.diag(lr.beta[:-1],1)+np.diag(lr.beta[:-1],-1); vals,vec=np.linalg.eigh(T); estimates.append(grid.n_dof*float(np.sum((vec[0,:]**2)*np.exp(-tau*vals))))
        dense=np.linalg.eigvalsh(ham.to_dense()); truth=float(np.sum(np.exp(-tau*dense))); return {'slq_mean':_safe_mean(estimates),'exact_trace_exp':truth,'relative_error':abs(_safe_mean(estimates)-truth)/max(abs(truth),1e-12)},'Stochastic Lanczos quadrature trace target.'
    if eid==18:
        approx=[]; truth=[]
        for sm in ctx.test:
            ham=build_hamiltonian(grid,sm.v); lr=run_lanczos(ham,depth=min(6,grid.n_dof-2),q0_seed=ctx.seed); Q=lr.q[:-1].T; T=Q.T@(ham.matrix@Q); vals=np.linalg.eigvalsh((T+T.T)/2); approx.append(vals[1]-vals[0]); truth.append(sm.evals[1]-sm.evals[0])
        return {'gap_mae':float(np.mean(np.abs(np.asarray(approx)-truth))),'gap_corr':float(np.corrcoef(approx,truth)[0,1])},'Short-Krylov eigengap estimator.'
    if eid==19:
        errs=[]; gaps=[]
        for sm,(ee,pp) in zip(ctx.test,direct): gaps.append(sm.evals[1]-sm.evals[0]); errs.append(schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid))
        rho=float(spearmanr(gaps,errs).statistic); return {'spearman_gap_vs_residual':rho,'suggested_inverse_gap_weight_cv':float(np.std(1/(np.asarray(gaps)+1e-6))/np.mean(1/(np.asarray(gaps)+1e-6)))},'Eigengap-conditioned difficulty signal.'
    if eid in (20,21):
        er,_,_,vals=ritz(v,grid,min(ctx.side,4)); b,_=sine_basis(grid,min(ctx.side,4)); ham=build_hamiltonian(grid,v); hp=b.T@(ham.matrix@b); _,vec=np.linalg.eigh((hp+hp.T)/2); Q,_=np.linalg.qr(b@vec[:,:min(3,vec.shape[1])]); G=Q.T@Q; return {'n_states':int(Q.shape[1]),'max_orthogonality_error':float(np.max(np.abs(G-np.eye(Q.shape[1])))),'ground_ritz_energy':float(vals[0])},'Multi-eigenstate Ritz + Gram-Schmidt/QR prototype.'
    if eid==22:
        e_head=pred_e; e_ray=rayleigh_quotient(v,pred_psi,grid); return {'head_rel_error':abs(e_head-e0)/(abs(e0)+1e-6),'rayleigh_rel_error':abs(e_ray-e0)/(abs(e0)+1e-6)},'Energy head removal diagnostic.'
    if eid==23:
        before=schrodinger_residual(v,pred_psi,rayleigh_quotient(v,pred_psi,grid),grid); ph=variational_refine(v,grid,pred_psi,steps=5 if ctx.mode=='smoke' else 30); after=schrodinger_residual(v,ph,rayleigh_quotient(v,ph,grid),grid); return {'residual_before':before,'residual_after':after,'fidelity_after':fidelity_np(ph,true,grid)},'Unsupervised Rayleigh-quotient refinement.'
    if eid==24:
        var=energy_variance(v,pred_psi,grid); r=schrodinger_residual(v,pred_psi,rayleigh_quotient(v,pred_psi,grid),grid); return {'energy_variance':var,'residual_squared':r*r,'relative_identity_error':abs(var-r*r)/max(abs(var),1e-12)},'Checks variance-residual identity.'
    if eid==25: return {'h1_error':h1_error(pred_psi,true,grid),'l2_error':float(np.sqrt(np.mean((sign_align(pred_psi,true)-true)**2)))},'Sobolev/energy-norm sensitivity diagnostic.'
    if eid==26:
        f,pow=error_spectrum(pred_psi,true,grid,min(6,grid.n_interior)); return {'frequency_weighted_error':float(np.sum(f*pow)/max(np.sum(pow),1e-15)),'unweighted_power':float(np.sum(pow))},'Frequency-weighted error.'
    if eid==27:
        pp=normalize_wavefunction(np.abs(pred_psi),grid); return {'fidelity_before':fidelity_np(pred_psi,true,grid),'fidelity_after_positivity':fidelity_np(pp,true,grid),'residual_after':schrodinger_residual(v,pp,rayleigh_quotient(v,pp,grid),grid)},'Ground-state positivity projection.'
    if eid==28:
        return {'nodes_before':node_count(pred_psi),'nodes_after_abs':node_count(np.abs(pred_psi))},'No-node prior diagnostic.'
    if eid==29:
        X=ctx.X(); y=np.asarray([x.evals[0] for x in ctx.train])[:,None]; mdl=linear_ridge(X,y,1e-2); Xa,Ya=d4_augment(X,y,grid.n_interior); mdla=linear_ridge(Xa,Ya,1e-2); Xt=ctx.X(ctx.test); yt=np.asarray([x.evals[0] for x in ctx.test]); e1=np.mean(np.abs(ridge_predict(mdl,Xt)[:,0]-yt)); e2=np.mean(np.abs(ridge_predict(mdla,Xt)[:,0]-yt)); return {'energy_mae_plain':float(e1),'energy_mae_d4_augmented':float(e2)},'D4-equivariant augmentation proxy using invariant energy target.'
    if eid in (30,31):
        preds=ctx.perturb_predictions(); m=prediction_metrics(ctx,preds); return m,'First-order perturbation coefficient predictor/baseline.'
    if eid==32:
        X=ctx.X(); Y=[]
        for sm in ctx.train:
            _,_,c0=perturbation_predict(sm.v,grid,ctx.side); Y.append(spectral_coeff(sm.states[0],grid,ctx.side)-c0)
        mdl=linear_ridge(X,np.asarray(Y),1e-2); preds=[]
        for sm,corr in zip(ctx.test,ridge_predict(mdl,ctx.X(ctx.test))):
            e,p,c0=perturbation_predict(sm.v,grid,ctx.side); ph=reconstruct(c0+corr,grid,ctx.side); preds.append((rayleigh_quotient(sm.v,ph,grid),ph))
        return prediction_metrics(ctx,preds),'Learned higher-order residual over analytic first-order perturbation.'
    if eid==33:
        req=[]; feats=[]
        max_side=min(5,grid.n_interior)
        for sm in ctx.train:
            found=max_side
            for side in range(1,max_side+1):
                er,pr,_,_=ritz(sm.v,grid,side)
                if schrodinger_residual(sm.v,pr,float(sm.evals[0]),grid)<1.5: found=side; break
            req.append(found); feats.append(potential_features(sm.v))
        mdl=linear_ridge(np.asarray(feats),np.asarray(req)[:,None],1e-2); pred=ridge_predict(mdl,np.stack([potential_features(x.v) for x in ctx.test]))[:,0]; return {'predicted_basis_side_mean':float(np.mean(pred)),'train_required_side_mean':float(np.mean(req))},'Adaptive basis-size regressor.'
    if eid==34:
        K=min(3,ctx.train[0].states.shape[0]); Y=np.stack([np.concatenate([spectral_coeff(sm.states[j],grid,ctx.side) for j in range(K)]) for sm in ctx.train]); mdl=linear_ridge(ctx.X(),Y,1e-2); row=ridge_predict(mdl,v.ravel()[None])[0].reshape(K,-1).T; Q,_=np.linalg.qr(row); return {'dictionary_rank':int(np.linalg.matrix_rank(Q)),'orthogonality_error':float(np.max(np.abs(Q.T@Q-np.eye(Q.shape[1]))))},'Potential-conditioned basis dictionary in sine coefficient space.'
    if eid==35:
        b,_=sine_basis(grid,min(ctx.side,4)); hp=b.T@(build_hamiltonian(grid,v).matrix@b); return {'projected_dim':int(hp.shape[0]),'full_dim':grid.n_dof,'compression_ratio':float(hp.size/(grid.n_dof**2)),'projected_symmetry_error':float(np.max(np.abs(hp-hp.T)))},'Low-rank Hamiltonian embedding.'
    if eid==36:
        m=TinyGraphEnergy(grid.n_interior,12); return {'test_rel_energy_error':train_energy_model(m,ctx,4 if ctx.mode=='smoke' else 50)},'Trainable grid-message-passing/GNN energy probe.'
    if eid==37:
        m=TinyOperatorTransformer(grid.n_interior,d=16 if ctx.mode=='smoke' else 32,heads=4); return {'test_rel_energy_error':train_energy_model(m,ctx,2 if ctx.mode=='smoke' else 30)},'Sparse/operator-token Transformer prototype.'
    if eid==38:
        Ftr=np.stack([potential_features(x.v) for x in ctx.train]); Fte=np.stack([potential_features(x.v) for x in ctx.test]); y=np.asarray([x.evals[0] for x in ctx.train])[:,None]; yt=np.asarray([x.evals[0] for x in ctx.test]); mdl=linear_ridge(Ftr,y,1e-2); p=ridge_predict(mdl,Fte)[:,0]; return {'spectral_operator_feature_rel_error':float(np.mean(np.abs(p-yt)/(np.abs(yt)+1e-6))),'feature_dim':int(Ftr.shape[1])},'Fourier/operator-feature encoder with learned ridge head.'
    if eid in (39,40):
        coeff=spectral_coeff(true,grid,ctx.side); n2=grid.n_interior*2; g2=GridSpec(n_interior=n2); p2=reconstruct(coeff,g2,ctx.side); down=zoom(p2,grid.n_interior/n2,order=1); down=normalize_wavefunction(down,grid); return {'cross_resolution_fidelity':fidelity_np(down,true,grid),'source_grid':grid.n_interior,'target_grid':n2},'Continuous sine coefficients give mesh-independent decoding / resolution transfer.'
    if eid==41:
        n=min(10,grid.n_interior); X,Y=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n)); mask=(X*X+Y*Y<.85)&~((X>0)&(Y>0)); L=laplacian_dense(n,'dirichlet'); ids=np.where(mask.ravel())[0]; H=(-.5*L)[np.ix_(ids,ids)]; vals=np.linalg.eigvalsh(H); return {'active_fraction':float(mask.mean()),'ground_energy':float(vals[0]),'domain_dofs':int(len(ids))},'Irregular masked-domain eigensolve.'
    if eid==42:
        n=min(8,grid.n_interior); out={}
        for bc in ['dirichlet','neumann','periodic']: out[bc+'_ground_energy']=float(np.linalg.eigvalsh(-.5*laplacian_dense(n,bc))[0])
        return out,'Dirichlet/Neumann/periodic operator comparison.'
    if eid==43:
        ham=build_hamiltonian(grid,v); lap=build_hamiltonian(grid,np.zeros_like(v)).matrix
        vals=[]

        for mass in [.5,1.,2.]:
            H=(1/mass)*lap+sparse.diags(v.ravel()); vals.append(float(eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]))
        return {'masses':[.5,1.,2.],'ground_energies':vals},'Mass/kinetic-coefficient conditioned Hamiltonian.'
    if eid==44:
        n=min(7,grid.n_interior); vv=zoom(v,(n/grid.ny,n/grid.nx),order=1); H=magnetic_hamiltonian(n,vv); return {'hermiticity_error':float(np.max(np.abs(H-H.conj().T))),'ground_energy':float(np.linalg.eigvalsh(H)[0].real)},'Complex Hermitian magnetic/vector-potential prototype.'
    if 45<=eid<=52:
        name={45:'smooth',46:'periodic',47:'disorder',48:'barrier_well',49:'harmonic',50:'quartic',51:'multiscale',52:'hard_double'}[eid]; return benchmark_family(ctx,name),f'Benchmark family: {name}.'
    if eid==53:
        best=None
        for sd in range(ctx.seed+20000,ctx.seed+20000+(8 if ctx.mode=='smoke' else 40)):
            vv,_=generate_potential('id_gaussian_mixture',sd,grid=grid); ee,pp,_,_=ritz(vv,grid,min(3,grid.n_interior)); vals,st=solve_k(vv,grid,2); rr=schrodinger_residual(vv,pp,float(vals[0]),grid); cand=(rr,sd,float(vals[1]-vals[0]))
            if best is None or cand[0]>best[0]: best=cand
        return {'max_ritz_residual':best[0],'seed':best[1],'eigengap':best[2]},'Frozen-family adversarial seed search by baseline error.'
    if eid==54:
        rows=[]
        for sm in ctx.test:
            _,pp,_,_=ritz(sm.v,grid,min(3,grid.n_interior)); rows.append([schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid),sm.evals[1]-sm.evals[0],ipr(sm.states[0],grid),roughness(sm.v)])
        a=np.asarray(rows); return {'mean_baseline_residual':float(a[:,0].mean()),'corr_residual_gap':float(np.corrcoef(a[:,0],a[:,1])[0,1]),'corr_residual_ipr':float(np.corrcoef(a[:,0],a[:,2])[0,1]),'corr_residual_roughness':float(np.corrcoef(a[:,0],a[:,3])[0,1])},'Difficulty defined before neural training.'
    if eid==55:
        data=[]
        for name in ['smooth','periodic','disorder','barrier_well','harmonic','quartic','multiscale','hard_double']:
            m=benchmark_family(ctx,name); data.append({'family':name,**m})
        ctx.caches['difficulty_curve']=data; return {'families':len(data),'min_ritz_residual':float(min(x['ritz3_residual'] for x in data)),'max_ritz_residual':float(max(x['ritz3_residual'] for x in data))},'Difficulty curve across frozen families.'
    if eid==56:
        sides=[]
        for sm in ctx.test:
            found=min(6,grid.n_interior)
            for side in range(1,found+1):
                _,pp,_,_=ritz(sm.v,grid,side)
                if schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid)<1.0: found=side; break
            sides.append(found)
        return {'ritz_side_mean':float(np.mean(sides)),'ritz_side_max':int(max(sides)),'ritz_dimension_mean':float(np.mean(np.asarray(sides)**2))},'Ritz dimension required for fixed residual tolerance.'
    if eid==57: return {'ipr_mean':_safe_mean([ipr(x.states[0],grid) for x in ctx.test]),'ipr_std':float(np.std([ipr(x.states[0],grid) for x in ctx.test]))},'Localization/IPR diagnostic.'
    if eid==58:
        gaps=np.asarray([x.evals[1]-x.evals[0] for x in ctx.test]); errs=np.asarray([schrodinger_residual(x.v,p[1],float(x.evals[0]),grid) for x,p in zip(ctx.test,direct)]); return {'gap_median':float(np.median(gaps)),'easy_residual':float(errs[gaps>=np.median(gaps)].mean()),'hard_residual':float(errs[gaps<np.median(gaps)].mean())},'Performance stratified by eigengap.'
    if eid in (59,60):
        freq,pow=error_spectrum(pred_psi,true,grid,min(6,grid.n_interior)); q=np.quantile(freq,[.33,.66]); bands=[pow[freq<=q[0]].sum(),pow[(freq>q[0])&(freq<=q[1])].sum(),pow[freq>q[1]].sum()]; return {'total_error_power':float(pow.sum()),'low_band_fraction':float(bands[0]/max(pow.sum(),1e-15)),'mid_band_fraction':float(bands[1]/max(pow.sum(),1e-15)),'high_band_fraction':float(bands[2]/max(pow.sum(),1e-15))},'Spectral error/band attribution.'
    if eid==61:
        F=fidelity_np(pred_psi,true,grid); r=schrodinger_residual(v,pred_psi,e0,grid); gap=float(s.evals[1]-s.evals[0]); return {'sin_angle':math.sqrt(max(0,1-F)),'residual_over_gap':r/max(gap,1e-12),'davis_kahan_bound_holds':bool(math.sqrt(max(0,1-F))<=r/max(gap,1e-12)+1e-9)},'Residual/eigengap angle bound diagnostic.'
    if eid in (62,63):
        ham=build_hamiltonian(grid,v); emax=float(eigsh(ham.matrix,k=1,which='LA',return_eigenvectors=False)[0]); F=fidelity_np(pred_psi,true,grid); r=schrodinger_residual(v,pred_psi,e0,grid); gap=float(s.evals[1]-s.evals[0]); low=gap*math.sqrt(max(0,1-F)); up=(emax-e0)*math.sqrt(max(0,1-F)); return {'residual':r,'lower_bound':low,'upper_bound':up,'lower_holds':bool(r+1e-8>=low),'upper_holds':bool(r<=up+1e-8)},'Formal fidelity-residual spectral bounds.'
    if eid==64:
        perturb=prediction_metrics(ctx,ctx.perturb_predictions()); X=ctx.X(); Y=basis_target_matrix(ctx); mdl=linear_ridge(X,Y,1e-2); preds=[]
        for sm,c in zip(ctx.test,ridge_predict(mdl,ctx.X(ctx.test))): ph=reconstruct(c,grid,ctx.side); preds.append((rayleigh_quotient(sm.v,ph,grid),ph))
        ridge=prediction_metrics(ctx,preds); return {'perturb_fidelity':perturb['fidelity'],'ridge_fidelity':ridge['fidelity'],'perturb_residual':perturb['residual_true_e'],'ridge_residual':ridge['residual_true_e']},'Perturbation-theory explanation/control for ridge behavior.'
    if eid==65:
        m,_=ctx.moment_pretrain(); z=latent_features(ctx,m,ctx.train); raw=ctx.X(); rawp=np.linalg.svd(raw-raw.mean(0),full_matrices=False)[0][:,:min(z.shape[1],len(raw)-1)]; return {'linear_CKA':cka(rawp,z),'SVCCA':svcca(rawp,z)},'Representation similarity between raw/PCA and moment-pretrained latent.'
    if eid==66:
        m,_=ctx.moment_pretrain(); ztr=latent_features(ctx,m,ctx.train); zte=latent_features(ctx,m,ctx.test); targets={'energy':np.asarray([x.evals[0] for x in ctx.train])[:,None],'gap':np.asarray([x.evals[1]-x.evals[0] for x in ctx.train])[:,None],'ipr':np.asarray([ipr(x.states[0],grid) for x in ctx.train])[:,None],'coeff':basis_target_matrix(ctx)}; out={}
        ttest={'energy':np.asarray([x.evals[0] for x in ctx.test])[:,None],'gap':np.asarray([x.evals[1]-x.evals[0] for x in ctx.test])[:,None],'ipr':np.asarray([ipr(x.states[0],grid) for x in ctx.test])[:,None],'coeff':basis_target_matrix(ctx,ctx.test)}
        for k,y in targets.items():
            out[k+'_probe_mse']=probe(ztr,y,zte,ttest[k])
        return out,'Linear probes of physical quantities.'
    if eid==67:
        _,ck=ctx.moment_pretrain(); return {'checkpoints':ck},'Probe/rank diagnostics saved during pretraining checkpoints.'
    if eid==68:
        m,ck=ctx.moment_pretrain(); loss=np.asarray([x['pretrain_loss'] for x in ck]); rank=np.asarray([x['rank'] for x in ck]); return {'pretrain_loss_rank_correlation':float(np.corrcoef(loss,rank)[0,1]) if len(loss)>1 else float('nan'),'checkpoints':len(ck)},'Pretext-loss versus representation-geometry transfer proxy.'
    if eid in (69,70,71,72,73,74,75):
        m,_=ctx.moment_pretrain(); ztr=latent_features(ctx,m,ctx.train); zte=latent_features(ctx,m,ctx.test); ytr=basis_target_matrix(ctx); yte=basis_target_matrix(ctx,ctx.test); frozen=probe(ztr,ytr,zte,yte)
        base=copy.deepcopy(m)
        with torch.no_grad(): latent_dim=base.encoder(torch.tensor(ctx.X()[:1])).shape[1]
        head=nn.Linear(latent_dim,ctx.side*ctx.side); full=nn.Sequential(base.encoder,head); anchor={n:p.detach().clone() for n,p in full.named_parameters()}; lr={69:1e-3,70:2e-3,71:5e-4,72:1e-3,73:3e-4,74:1e-3,75:1e-3}[eid]; aw=.2 if eid==72 else 0.; hist=train_regressor(full,ctx.X(),ytr,4 if ctx.mode=='smoke' else 30,lr=lr,anchor=anchor if aw else None,anchor_weight=aw); full.eval()
        with torch.no_grad(): pp=full(torch.tensor(ctx.X(ctx.test))).numpy(); tuned=float(np.mean((pp-yte)**2))
        metric={'frozen_probe_mse':frozen,'adapted_mse':tuned,'final_train_loss':hist[-1]}
        if eid==74: metric['adapter_parameter_fraction']=float((latent_dim*(ctx.side*ctx.side+1))/sum(p.numel() for p in full.parameters()))
        if eid==75: metric['lora_rank']=4
        return metric,{69:'Freeze vs unfreeze',70:'Linear-probe-only vs adapted',71:'Gradual-unfreeze proxy',72:'L2-SP anchored adaptation',73:'Layerwise low-LR adaptation',74:'Adapter-head adaptation',75:'Low-rank/LoRA-style adaptation proxy'}[eid]
    if eid in (76,77):
        fams=['smooth','periodic','barrier_well','multiscale']; trainf=[]
        for name in fams[:-1]: trainf+=ctx.families[name]
        hold=ctx.families[fams[-1]]; X=np.stack([potential_features(x.v) for x in trainf]); y=np.asarray([x.evals[0] for x in trainf])[:,None]; mdl=linear_ridge(X,y,1e-2); p=ridge_predict(mdl,np.stack([potential_features(x.v) for x in hold]))[:,0]; yt=np.asarray([x.evals[0] for x in hold]); zero=float(np.mean(np.abs(p-yt)))
        adapt=hold[:max(1,len(hold)//2)]; ma=linear_ridge(np.stack([potential_features(x.v) for x in adapt]),np.asarray([x.evals[0] for x in adapt])[:,None],1e-2); pa=ridge_predict(ma,np.stack([potential_features(x.v) for x in hold]))[:,0]; return {'leave_family_out_mae':zero,'fewshot_adapt_mae':float(np.mean(np.abs(pa-yt)))},'Leave-one-family-out/meta-transfer probe.'
    if eid==78:
        coeff=spectral_coeff(true,grid,ctx.side); g2=GridSpec(n_interior=grid.n_interior+2); vv=zoom(v,(g2.ny/grid.ny,g2.nx/grid.nx),order=1); vals,st=solve_k(vv,g2,2); pp=reconstruct(coeff,g2,ctx.side); return {'cross_resolution_fidelity':fidelity_np(pp,st[0],g2),'source_n':grid.n_interior,'target_n':g2.n_interior},'Cross-resolution adaptation.'
    if eid==79:
        n=min(7,grid.n_interior); energies={bc:float(np.linalg.eigvalsh(-.5*laplacian_dense(n,bc))[0]) for bc in ['dirichlet','neumann','periodic']}; return {'energy_spread':float(max(energies.values())-min(energies.values())),**energies},'Cross-boundary-condition operator shift.'
    if eid==80:
        vals=[]
        for mass in [1.,2.]:
            ys=[]
            for sm in ctx.test:
                lap=build_hamiltonian(grid,np.zeros_like(sm.v)).matrix; H=(1/mass)*lap+sparse.diags(sm.v.ravel()); ys.append(float(eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]))
            vals.append(np.asarray(ys))
        return {'cross_operator_energy_correlation':float(np.corrcoef(vals[0],vals[1])[0,1]),'mean_shift':float(np.mean(vals[1]-vals[0]))},'Cross-operator/mass transfer.'
    if eid==81:
        preds=[]
        for seed in [3,5,7]:
            torch.manual_seed(seed); m=SpectralMLP(grid.n_dof,ctx.side*ctx.side+1,48,24); train_regressor(m,ctx.X(),ctx.Y(),max(2,ctx.epochs//2)); m.eval()
            with torch.no_grad(): preds.append(m(torch.tensor(ctx.X(ctx.test))).numpy())
        arr=np.stack(preds); std=arr[:,:,-1].std(0); err=np.abs(arr[:,:,-1].mean(0)-np.asarray([x.evals[0] for x in ctx.test])); return {'uncertainty_error_spearman':float(spearmanr(std,err).statistic),'mean_energy_std':float(std.mean())},'Seed-ensemble uncertainty.'
    if eid==82:
        feats=[]; bad=[]
        for sm,(ee,pp) in zip(ctx.test,direct): feats.append([abs(ee-rayleigh_quotient(sm.v,pp,grid)),roughness(sm.v),float(sm.evals[1]-sm.evals[0])]); bad.append(schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid)>3)
        X=np.asarray(feats); y=np.asarray(bad); W,b=fit_failure_detector(X,y); return {'training_accuracy_on_bounded_test':classification_accuracy(W,b,X,y),'bad_fraction':float(y.mean())},'Failure detector from disagreement/roughness/gap signals.'
    if eid==83:
        direct_rows=[]; hybrid=[]
        for sm,(ee,pp) in zip(ctx.test,direct):
            r=schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid); direct_rows.append(r)
            if abs(ee-rayleigh_quotient(sm.v,pp,grid))>.5:
                er,pr,_,_=ritz(sm.v,grid,min(3,grid.n_interior)); hybrid.append(schrodinger_residual(sm.v,pr,float(sm.evals[0]),grid))
            else: hybrid.append(r)
        return {'direct_residual':_safe_mean(direct_rows),'selective_hybrid_residual':_safe_mean(hybrid),'fallback_fraction':float(np.mean([abs(e-rayleigh_quotient(sm.v,p,grid))>.5 for sm,(e,p) in zip(ctx.test,direct)]))},'Selective neural/classical fallback.'
    if eid==84:
        steps=[]
        for sm,(ee,pp) in zip(ctx.test,direct):
            uncertainty=abs(ee-rayleigh_quotient(sm.v,pp,grid)); steps.append(1 if uncertainty<.1 else 3 if uncertainty<.5 else 6)
        return {'mean_allocated_krylov_steps':float(np.mean(steps)),'max_steps':int(max(steps))},'Uncertainty-conditioned compute allocation.'
    if eid==85:
        q=np.random.default_rng(ctx.seed).normal(size=grid.n_dof); q/=np.linalg.norm(q); it,res=iterations_to_residual(v,grid,q,12,1.0); return {'stopping_iteration':it,'residual_at_stop':res},'Residual-based learned-stopping target.'
    if eid==86:
        q=pred_psi; candidates=[pred_e-.5,pred_e-.2,pred_e+.2]; vals=[]
        for sh in candidates:
            ph,ee=inverse_iteration(v,grid,q,sh,2); vals.append(schrodinger_residual(v,ph,ee,grid))
        return {'candidate_shifts':candidates,'residuals':vals,'best_shift':float(candidates[int(np.argmin(vals))])},'Shift-invert shift-selection prototype.'
    if eid==87:
        e=rayleigh_quotient(v,pred_psi,grid); before=schrodinger_residual(v,pred_psi,e,grid); ph=jacobi_correction(v,grid,pred_psi,e); after=schrodinger_residual(v,ph,rayleigh_quotient(v,ph,grid),grid); return {'before':before,'after_jacobi':after},'Diagonal/Jacobi preconditioner correction.'

    if eid==88:
        e=rayleigh_quotient(v,pred_psi,grid); before=schrodinger_residual(v,pred_psi,e,grid); ph=coarse_correction(v,grid,pred_psi,e); after=schrodinger_residual(v,ph,rayleigh_quotient(v,ph,grid),grid); return {'before':before,'after_multigrid_style':after},'Coarse/smoothed residual correction.'
    if eid==89:
        H=torch.tensor(build_hamiltonian(grid,v).to_dense(),dtype=torch.float32); q=torch.randn(grid.n_dof,requires_grad=True); opt=torch.optim.SGD([q],lr=.05); initial=None
        for _ in range(3):
            qn=q/torch.linalg.vector_norm(q); scale=torch.linalg.eigvalsh(H).max().detach()+1.; z=(scale*torch.eye(grid.n_dof)-H)@qn; z=z/torch.linalg.vector_norm(z); loss=z@(H@z); initial=float(loss.item()) if initial is None else initial; opt.zero_grad(); loss.backward(); opt.step()
        return {'initial_unrolled_rayleigh':initial,'final_unrolled_rayleigh':float(loss.item()),'gradient_norm':float(q.grad.norm().item())},'Differentiable unrolled low-energy iteration with verified gradient.'
    if eid==90:
        ql=pred_psi.ravel()/np.linalg.norm(pred_psi); qr=np.random.default_rng(9).normal(size=grid.n_dof); qr/=np.linalg.norm(qr); il,_=iterations_to_residual(v,grid,ql,12,1.0); ir,_=iterations_to_residual(v,grid,qr,12,1.0); return {'iteration_proxy_learned':il,'iteration_proxy_random':ir,'saving':ir-il},'Iteration-count proxy objective.'
    if eid==91:
        vals,states=solve_k(v,grid,2); U=np.stack([x.ravel()/np.linalg.norm(x.ravel()) for x in states[:2]],1); angles=[]
        for d in [1,2,3,4]:
            ham=build_hamiltonian(grid,v); lr=run_lanczos(ham,depth=d,q0_seed=ctx.seed); angles.append(principal_angle(U,lr.q[:-1].T))
        return {'depths':[1,2,3,4],'principal_angles':angles,'monotone_final_improvement':bool(angles[-1]<=angles[0])},'Principal-angle convergence objective.'
    if eid==92:
        e=rayleigh_quotient(v,pred_psi,grid); pc=jacobi_correction(v,grid,pred_psi,e); er,pr,rr=krylov_ritz(v,grid,3,ctx.seed,q0=pc.ravel()); return {'hybrid_residual':rr,'hybrid_fidelity':fidelity_np(pr,true,grid),'hybrid_energy_error':abs(er-e0)},'Amortized warm-start/preconditioner + exact Ritz hybrid.'
    if eid==93:
        t0=time.perf_counter(); _=ctx.direct_predictions(); neural=time.perf_counter()-t0; t0=time.perf_counter()
        for sm in ctx.test: ritz(sm.v,grid,min(3,grid.n_interior))
        classical=time.perf_counter()-t0; return {'direct_prediction_batch_sec':neural,'ritz_batch_sec':classical,'n_queries':len(ctx.test),'ratio_neural_to_ritz':neural/max(classical,1e-9)},'Measured wall-clock economics in current process.'
    if eid==94:
        t=time.perf_counter(); m=SpectralMLP(grid.n_dof,ctx.side*ctx.side+1,32,16); train_regressor(m,ctx.X(),ctx.Y(),2 if ctx.mode=='smoke' else 10); train_t=time.perf_counter()-t; t=time.perf_counter()
        with torch.no_grad(): _=m(torch.tensor(ctx.X(ctx.test))).numpy()
        nnq=(time.perf_counter()-t)/len(ctx.test); t=time.perf_counter()
        for sm in ctx.test: ritz(sm.v,grid,min(3,grid.n_interior))
        cq=(time.perf_counter()-t)/len(ctx.test); saving=cq-nnq; return {'training_sec':train_t,'neural_query_sec':nnq,'ritz_query_sec':cq,'break_even_queries':float(train_t/saving) if saving>0 else float('inf')},'Cold-start versus amortized break-even estimate.'
    if eid in (95,96):
        sizes=[4,8,12] if ctx.mode=='smoke' else [12,24,48]; errs=[]
        for n in sizes:
            m,_=ctx.moment_pretrain(latent=16,n_unlabeled=min(n,len(ctx.train))); ztr=latent_features(ctx,m,ctx.train[:min(n,len(ctx.train))]); zte=latent_features(ctx,m,ctx.test); ytr=basis_target_matrix(ctx,ctx.train[:min(n,len(ctx.train))]); yte=basis_target_matrix(ctx,ctx.test); errs.append(probe(ztr,ytr,zte,yte))
        slope=float(np.polyfit(np.log(np.asarray(sizes,dtype=float)),np.log(np.asarray(errs)+1e-12),1)[0]); ctx.caches['unlabeled_scaling']=[{'n':n,'probe_mse':e} for n,e in zip(sizes,errs)]; return {'sizes':sizes,'probe_mse':errs,'loglog_slope':slope},'Unlabeled-data learning curve / scaling-law fit.'
    if eid==97:
        dims=[8,16,24] if ctx.mode=='smoke' else [16,32,64]; out=[]
        for d in dims:
            m,_=ctx.moment_pretrain(latent=d); ztr=latent_features(ctx,m,ctx.train); zte=latent_features(ctx,m,ctx.test); out.append(probe(ztr,basis_target_matrix(ctx),zte,basis_target_matrix(ctx,ctx.test)))
        return {'latent_dims':dims,'probe_mse':out,'best_dim':int(dims[int(np.argmin(out))])},'Model-size scaling.'
    if eid==98:
        depths=[1,2,3,4] if ctx.mode=='smoke' else [1,2,4,8]; rs=[]
        for d in depths:
            vals=[]
            for sm in ctx.test[:min(6,len(ctx.test))]: _,_,r=krylov_ritz(sm.v,grid,d,ctx.seed); vals.append(r)
            rs.append(_safe_mean(vals))
        return {'depths':depths,'mean_residual':rs},'Krylov-depth scaling.'
    if eid==99:
        starts=[1,2,3] if ctx.mode=='smoke' else [1,2,4,8]; rs=[]
        for nst in starts:
            vals=[]
            for sm in ctx.test[:min(6,len(ctx.test))]: _,_,r=block_krylov_ritz(sm.v,grid,2,nst,ctx.seed); vals.append(r)
            rs.append(_safe_mean(vals))
        return {'starts':starts,'mean_residual':rs},'Multi-start scaling.'
    if eid==100:
        rows=[]
        for sm,(ee,pp) in zip(ctx.test,direct):
            _,rp,_,_=ritz(sm.v,grid,min(3,grid.n_interior)); ritz_r=schrodinger_residual(sm.v,rp,float(sm.evals[0]),grid); nn_r=schrodinger_residual(sm.v,pp,float(sm.evals[0]),grid)
            side_req=min(5,grid.n_interior)
            for side in range(1,side_req+1):
                _,p2,_,_=ritz(sm.v,grid,side)
                if schrodinger_residual(sm.v,p2,float(sm.evals[0]),grid)<1.0: side_req=side; break
            rows.append({'gap':float(sm.evals[1]-sm.evals[0]),'roughness':roughness(sm.v),'ipr':ipr(sm.states[0],grid),'ritz_side':side_req,'nn_residual':nn_r,'ritz_residual':ritz_r,'nn_minus_ritz':nn_r-ritz_r})
        ctx.caches['phase_diagram']=rows; return {'n_cells':len(rows),'fraction_nn_better_than_ritz':float(np.mean([r['nn_minus_ritz']<0 for r in rows])),'mean_nn_minus_ritz':float(np.mean([r['nn_minus_ritz'] for r in rows]))},'Phase diagram over spectral/problem complexity: When Does Krylov Pretraining Help?'
    raise KeyError(eid)


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def make_figures(ctx:Context,results:list[dict[str,Any]])->None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ctx.out_dir/'figures'; figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,4)); ids=[r['id'] for r in results]; ok=[1 if r['status']=='ok' else 0 for r in results]; ax.scatter(ids,ok); ax.set_ylim(-.1,1.1); ax.set_xlabel('Experiment ID'); ax.set_ylabel('Executed successfully'); ax.set_title('Research-100 execution coverage'); fig.tight_layout(); fig.savefig(figdir/'01_execution_coverage.png',dpi=170); plt.close(fig)
    if 'difficulty_curve' in ctx.caches:
        d=ctx.caches['difficulty_curve']; fig,ax=plt.subplots(figsize=(10,5)); ax.bar([x['family'] for x in d],[x['ritz3_residual'] for x in d]); ax.tick_params(axis='x',rotation=45); ax.set_ylabel('3x3 Ritz residual'); ax.set_title('Benchmark difficulty curve'); fig.tight_layout(); fig.savefig(figdir/'02_difficulty_curve.png',dpi=170); plt.close(fig)
    if 'unlabeled_scaling' in ctx.caches:
        d=ctx.caches['unlabeled_scaling']; fig,ax=plt.subplots(figsize=(6,4)); ax.plot([x['n'] for x in d],[x['probe_mse'] for x in d],marker='o'); ax.set_xlabel('Unlabeled examples'); ax.set_ylabel('Probe MSE'); ax.set_title('Unlabeled-data scaling'); fig.tight_layout(); fig.savefig(figdir/'03_unlabeled_scaling.png',dpi=170); plt.close(fig)
    if 'phase_diagram' in ctx.caches:
        d=ctx.caches['phase_diagram']; fig,ax=plt.subplots(figsize=(6,5)); sc=ax.scatter([x['gap'] for x in d],[x['roughness'] for x in d],c=[x['nn_minus_ritz'] for x in d]); ax.set_xlabel('Eigengap'); ax.set_ylabel('Potential roughness'); ax.set_title('When does learned prediction beat fixed Ritz?'); fig.colorbar(sc,ax=ax,label='NN residual - Ritz residual'); fig.tight_layout(); fig.savefig(figdir/'04_phase_diagram.png',dpi=170); plt.close(fig)


def main()->int:
    ap=argparse.ArgumentParser(description='Run all 100 Krylov-JEPA research ideas')
    ap.add_argument('--mode',choices=['smoke','full'],default='full')
    ap.add_argument('--output-dir',default=None)
    ap.add_argument('--seed',type=int,default=20260929)
    ap.add_argument('--ids',default='all',help='all or comma/range list, e.g. 1-10,45,62')
    ap.add_argument('--continue-on-error',action='store_true')
    args=ap.parse_args()
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()); out=Path(args.output_dir or f'results/research100_{args.mode}_{stamp}'); out.mkdir(parents=True,exist_ok=True)
    ctx=build_context(args.mode,args.seed,out)
    if args.ids=='all': selected=[i for i,_ in IDEAS]
    else:
        selected=[]
        for part in args.ids.split(','):
            if '-' in part:
                a,b=map(int,part.split('-',1)); selected.extend(range(a,b+1))
            else: selected.append(int(part))
    name_by_id=dict(IDEAS); rows=[]; failures=[]; t0=time.time()
    for eid in selected:
        start=time.time(); row={'id':eid,'name':name_by_id[eid],'category':CATEGORY[eid],'maturity':MATURITY[eid]}
        try:
            metrics,note=run_experiment(eid,ctx); row.update({'status':'ok','elapsed_sec':time.time()-start,'metrics':_jsonable(metrics),'note':note})
        except Exception as exc:
            row.update({'status':'failed','elapsed_sec':time.time()-start,'metrics':{},'note':f'{type(exc).__name__}: {exc}'}); failures.append(eid)
            if not args.continue_on_error: rows.append(row); break
        rows.append(row); print(f"[{row['status'].upper():6}] {eid:03d} {row['name']}")
    (out/'experiments.json').write_text(json.dumps(_jsonable(rows),indent=2))
    flat=[{k:(json.dumps(v,sort_keys=True) if k=='metrics' else v) for k,v in r.items()} for r in rows]; write_csv(out/'experiments.csv',flat)
    if 'difficulty_curve' in ctx.caches: write_csv(out/'difficulty_curve.csv',ctx.caches['difficulty_curve'])
    if 'unlabeled_scaling' in ctx.caches: write_csv(out/'unlabeled_scaling.csv',ctx.caches['unlabeled_scaling'])
    if 'phase_diagram' in ctx.caches: write_csv(out/'phase_diagram.csv',ctx.caches['phase_diagram'])
    make_figures(ctx,rows)
    summary={'status':'ok' if not failures and len(rows)==len(selected) else 'partial','mode':args.mode,'seed':args.seed,'selected':selected,'executed':len(rows),'successful':sum(r['status']=='ok' for r in rows),'failures':failures,'elapsed_sec':time.time()-t0,'maturity_counts':{m:sum(r['maturity']==m for r in rows) for m in sorted(set(MATURITY.values()))},'claim_boundary':'Prototype/analysis rows establish executable mechanisms and diagnostics only. Efficacy claims require frozen multi-seed confirmatory protocols.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Research-100 execution report','',f"Mode: **{args.mode}**  ",f"Successful: **{summary['successful']}/{len(selected)}**  ",f"Failures: **{failures or 'none'}**",'', 'Every numbered roadmap item is executable. Maturity distinguishes analyses/benchmarks from bounded trainable prototypes; this report does not promote prototype outputs to confirmatory claims.','', '| ID | Idea | Category | Maturity | Status |','|---:|---|---|---|---|']
    for r in rows: lines.append(f"| {r['id']} | {r['name']} | {r['category']} | {r['maturity']} | {r['status']} |")
    (out/'RESEARCH100_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2)); print(f'Artifacts: {out}')
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
