# Protocol Amendment 0004 — Preserve SSL features (adaptation breakthrough)

**Date:** 2026-09-26  
**Status:** ACTIVE  
**Study:** 3  

## Problem (from Study-2 autopsy)

Study-2 fixed corpus mismatch and shuffle semantics, but full / early-unfreeze fine-tuning **erased** LMOP’s advantage:

- Zero-shot genuine test_id relative \(L_2\): LMOP **0.69** < MML **0.78** (LMOP better).
- After Study-2 fine-tune: MML wins every cell.
- Pretrain loss mix: \(\lambda_{\mathrm{var}}=25\) contributed ~77% of LMOP loss — VICReg drowned JEPA/hybrid signal.

## Amendment (frozen a priori)

### A. Loss rebalance

- \(\lambda_{\mathrm{var}} = 1.0\) (was 25)
- \(\lambda_{\mathrm{cov}} = 1.0\) (unchanged)
- \(\lambda_u = 1.0\) (was 0.1) — hybrid head matched to MML scale
- \(\gamma = 1.0\) unchanged

### B. Primary adaptation protocols (decisive)

Compare Scratch / MML / LMOP under **matched** adaptation:

1. **`zero_shot`** — no genuine fine-tune (pretrained head only; Scratch = random init).
2. **`probe`** — freeze backbone for **all** fine-tune epochs; train `proj` head only.
3. **`low_lr_ft`** (secondary) — unfreeze backbone at `lr × backbone_lr_scale` with `backbone_lr_scale=0.05`.

Study-2 full FT (`freeze_epochs` then full `lr`) remains an **archive ablation**, not the primary claim gate.

### C. Claim support (Study-3)

Support if LMOP beats MML on relative \(L_2\) with non-overlapping seed CIs on **at least one primary protocol** among `{zero_shot, probe}` at \(N\in\{25,100\}\) ID and/or OOD, **and** shuffled pretrain loss > correct (reuse Study-2 / Study-3 shuffle).

### D. Data / PDE

Reuse Study-2 `data/confirmatory_v2/` and `data/smoke_v2/` (matched MMS). No corpus redesign.

### E. Optional architecture note

Separate \(E_u\) tower deferred unless probe still fails after A–B.

## Invalidates for Study-3 primary claims

- Study-2 full-FT tables as evidence that “LMOP cannot win”
- \(\lambda_{\mathrm{var}}=25\) recipe

Study-1 and Study-2 remain archived negatives under their respective adaptation protocols.
