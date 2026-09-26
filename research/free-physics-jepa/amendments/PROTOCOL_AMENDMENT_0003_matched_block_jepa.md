# Protocol Amendment 0003 — Matched MMS + same-encoder block-JEPA

**Date:** 2026-09-26  
**Status:** ACTIVE  
**Supersedes for new runs:** Study-1 confirmatory (`confirmatory_20260926T150845Z`) remains archived as a **negative / null result** under amendments 0001–0002. This amendment defines **Study-2**.

## Problem (from Study-1 autopsy)

1. Manufactured MIXED solutions were **7.4×** larger in \(\|u\|_2\) and broadband (≈31% low-frequency energy) vs genuine \(f\equiv 1\) Darcy (≈99% low-frequency). Neither MML nor LMOP beat Scratch; methods tied.
2. LMOP used a **3-channel** context encoder and a **solution-only** target view `(u,u,0)`, then transferred only partial lift weights into a 2-channel downstream — weak inductive alignment.
3. Masking was pixel-wise salt; I-JEPA evidence favors **multi-block** targets.
4. Shuffled control kept `(f,u)` paired and only swapped `a`, so the control could be *easier* than correct physics (observed: shuffle loss 0.78 < correct 0.82).
5. Full fine-tune washed out any init by epoch ~3–4.

## Amendment (frozen a priori before Study-2 outcomes)

### A. Matched manufactured corpus (still solver-free MMS)

1. Primary family renamed **`matched`**: sine-mode MMS with `kmax ≤ 4`, `n_modes ∈ [1,3]`.
2. After assembling \(u\), **rescale** so \(\|u\|_2\) is drawn uniformly from `[u_norm_min, u_norm_max] = [1.8, 3.4]` (covers genuine train norm≈2.6), then set \(f = A_a u\).
3. Ablation family `low` unchanged except it also applies the same norm matching.
4. Data live under `data/{smoke,confirmatory}_v2/` (Study-1 HDF5 retained, not reused).

### B. Per-instance input normalization

For every train/eval example:
- \(a \leftarrow a / (\mathrm{mean}(a)+\epsilon)\)
- \(f \leftarrow f / (\mathrm{rms}(f)+\epsilon)\)
- Targets \(u\) stay in physical units; loss remains relative MSE (scale-aware).
- Physics metrics on eval use **denormalized** predictions in physical \(u\) (model outputs physical \(u\) directly).

### C. Same-encoder multi-block LMOP-JEPA

1. Online encoder = **identical** `FNO2d(in=2)` as `DownstreamSolver.backbone`.
2. Target encoder = EMA copy of online; target view = `(u_n, u_n)` with \(u_n = u/(\mathrm{rms}(u)+\epsilon)\) stacked to 2 channels (stop-grad).
3. Context view = normalized `(a,f)` with **multi-block spatial mask** (1–3 blocks, ~15–40% area, zero-fill); loss only on masked spatial sites after LayerNorm.
4. Predictor: 3-layer 1×1 Conv bottleneck (width → width).
5. VICReg: variance hinge **and** covariance off-diagonal penalty on batch×channel pooled predictor outputs (`λ_var=25`, `λ_cov=1`, `γ=1`).
6. Optional hybrid head (frozen): `λ_u=0.1` relative MSE from online `proj(encode(a,f))` → `u` during pretrain.
7. Transfer: **full** `state_dict` copy online → downstream backbone (no partial lift hack).
8. Collapse kill: mean channel std \(<10^{-4}\) after 200 steps still aborts.

### D. Shuffled-physics control (fixed)

- Mode **`u_vs_af`**: keep `(a,f)` from index \(i\); replace `u` with `u[π(i)]` (breaks PDE consistency; does **not** preserve `(f,u)` pairing).
- Claim support requires shuffled pretrain loss **worse** than correct.

### E. Few-shot fine-tune schedule

- Epochs `[0, freeze_epochs)`: freeze backbone, train `proj` head only (`freeze_epochs=10` confirmatory / `5` smoke).
- Remaining epochs: unfreeze all at `lr`.
- Does not change budgets \(N\in\{25,100\}\), seeds `{11,23,47}`, metrics, or OOD definition.

### F. Compute

- Confirmatory SSL steps: **4000** (was 2000).
- Smoke SSL steps: **200** (was 100).

## Does not change

PDE operator, genuine generation, primary OOD (correlation-length), label budgets, seeds, primary metric (relative \(L_2\)), bootstrap settings, decisive comparison (MML vs LMOP).

## Invalidates for Study-2 claims

- Study-1 smoke/confirmatory checkpoints and tables (keep as archive / negative result).
- Amendment 0002 hinge-only recipe (replaced by full VICReg + block JEPA here).
- Old shuffle semantics.

## Claim gate (unchanged logic)

Support only if LMOP beats MML on relative \(L_2\) with non-overlapping seed CIs at \(N=25\) and/or \(N=100\) **and** fixed shuffle control is worse than correct physics.
