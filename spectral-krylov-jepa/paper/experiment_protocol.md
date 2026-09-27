# Frozen Experiment Protocol

**Freeze date:** 2026-09-26  
**Rule:** After freeze, changes driven by test-set outcomes must be labeled *exploratory* and must not silently rewrite this document.

## Physics

- Domain: \([0,1]^2\), Dirichlet \(\psi=0\) on \(\partial\Omega\)
- Default grid: \(32\times 32\) interior (`n_interior=32`), \(h=1/33\)
- Smoke grid: \(16\times 16\) interior
- Discretization: 5-point FD Laplacian; \(H=-\frac12\Delta + \mathrm{diag}(V)\)
- Eigensolver: `eigsh`, `which='SA'`, `tol=1e-8`, reject if relative residual \(>10^{-5}\)
- Lanczos depth default \(K=3\) with full reorthogonalization for short trajectories

## Potentials

**ID** (`id_gaussian_mixture`): 1–3 wells; \(A\in[-8,-2]\); \(\sigma\in[0.08,0.18]\); margin 0.15  

**OOD**

- narrow: \(\sigma\in[0.035,0.055]\)
- strong: \(A\in[-18,-12]\)
- double: 2 wells, min separation 0.45

## Data sizes

| Setting | Unlabeled | Labeled layout |
|---------|-----------|----------------|
| Smoke | 80 | train 24 / val 8 / test_ID 8 / OOD 6 each |
| Decisive (target) | 1,000 (optionally 5,000) | train 120 / val 40 / test_ID 40 / OOD 40 each |

Fine-tune subsets (nested): 10 ⊂ 25 ⊂ 50 ⊂ 100 (smoke: 10 only if train≥10).

Split seed: `2026`. Never modify test after seeing outcomes.

## Models

- Smoke: `embed_dim=64`, depth 2, 4 heads, mlp_ratio 2
- Default: `embed_dim=128`, depth 4, 4 heads, mlp_ratio 4, patch 4×4
- Target ~1–5M parameters for default

## Training

**Pretrain:** AdamW lr \(3\times10^{-4}\), wd 0.05, EMA 0.996, cosine+warmup, batch 8 (smoke 4), max_steps 2000 (smoke 40)

**Finetune:** AdamW lr \(10^{-3}\) (encoder \(0.3\times\) when pretrained), wd 0.01, epochs 80 (smoke 25), early stop patience 15 on selection score \((1-F)+0.5\cdot\mathrm{relE}\)

**Downstream loss (a priori):** \(\lambda_\psi=1.0\), \(\lambda_E=1.0\), \(\lambda_{\psi\mathrm{mse}}=0.1\) on
\(L=\lambda_\psi(1-F)+\lambda_{\psi\mathrm{mse}}\mathrm{signMSE}+\lambda_E\,\mathrm{zscoreMSE}(E)\)

Energy and potential inputs are standardized from the **fine-tune subset only** (no test leakage).

**Seeds:** 11, 23, 47 (smoke uses 11)

## Metrics

Primary: fidelity \(F=|\langle\hat\psi,\psi\rangle|^2\)  
Secondary:
- relative energy error (energy head)
- relative Rayleigh energy error \(E_R=\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle\)
- residual at \(\hat E\) (protocol)
- residual at true \(E_0\) (ψ quality)
- residual at Rayleigh \(E_R\) (eigenvector quality)

Bootstrap: 2000 paired resamples over test potentials (smoke may use 500)

## Ablations (declared)

Krylov depth K1/K2/K3; shuffled physics; remove V; coeff loss on/off; Field vs Operator vs Krylov

## Kill / document criteria

See project brief §34. Null results are valid scientific outcomes.
