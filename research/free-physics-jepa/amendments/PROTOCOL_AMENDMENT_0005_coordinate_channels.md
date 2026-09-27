# Protocol Amendment 0005 — Coordinate channels (downstream architecture repair)

**Date:** 2026-09-27  
**Status:** PROPOSED — requires author approval before any run. Not active; `protocol.yaml` is unchanged.  
**Study:** 4 (if approved)

## Problem (engineering audit, 2026-09-27)

`FNO2d` takes only `(a_n, f_n)`, has no coordinate channels and no padding, so every model in Studies 1–3 is exactly equivariant to periodic translations. Genuine data use \(f\equiv 1\), so after normalization the only informative input is a stationary random field; the model cannot locate the zero-Dirichlet boundary, while \(u\) is a boundary-pinned bump.

Evidence (all from existing artifacts; no test-set tuning):

| Check | Artifact | Result |
|---|---|---|
| Boundary/center magnitude of predictions | `results/tables/audit_positional_floor.json` | models 0.96–1.00 vs truth 0.06 |
| Trained models vs train-mean field (test ID / OOD) | same | 0.50–0.54 vs **0.33 / 0.24** |
| Floor for spatially constant output | same | 0.55 / 0.53 (models sit at this floor) |
| Scratch N=100, validation only, ± coordinate channels | `results/tables/audit_coord_sanity.json` | **0.509 → 0.084** (mean field 0.331) |
| Scale identifiability of normalized inputs (ruled out) | `results/tables/audit_scale_identifiability.json` | floor ≈0.03; not the cause |

Consequence: Studies 1–3 compare pretraining objectives in a regime where no downstream model beats an input-blind baseline. Their MML-vs-LMOP gaps (≤0.03) are not informative about the research question in either direction.

## Proposed change (to be frozen before any Study-4 outcome is seen)

1. Append normalized coordinates \((x, y)\in[0,1]^2\) as two extra input channels to **every** `FNO2d` (downstream solver, MML-direct, LMOP online/target encoders; LMOP target view becomes `(u_n, u_n, x, y)`).
2. **Sanity gate (new, mandatory):** on the validation split, Scratch at N=100 must beat the train-mean-field baseline for every seed before confirmatory results are computed. Fail → stop and document.
3. Everything else frozen as in amendment 0004: data `confirmatory_v2`, splits/manifests, N ∈ {25, 100}, seeds {11, 23, 47}, λ_var = 1, λ_cov = 1, λ_u = 1, 4000 SSL steps, 60 fine-tune epochs, relative L2 primary metric, correlation-length OOD, `u_vs_af` shuffle control.
4. Adaptation protocols, fixed a priori: **primary** = `probe`, `low_lr_ft`; **secondary** = `full_ft` (Study-2 schedule); `zero_shot` = descriptive only (the across-seed CI is degenerate, see `aggregate_claim_gate.seed_variance_degenerate`).
5. Claim gate unchanged otherwise (non-overlapping across-seed CIs, all seeds favor LMOP, shuffled loss > correct), plus the sanity gate above.

## Forking-paths disclosure

This would be the fifth amendment within two days, each following an unfavorable result. Any Study-4 report must present Studies 1–3 alongside it, state that 0005 was motivated by a failed trivial-baseline check (not by the MML-vs-LMOP comparison), and must not pool results across studies.

## Estimated compute

Laptop MPS: ≈4–5 h (MML + LMOP + shuffled pretrain ≈ 30 min; 3 fine-tune protocols × 18 runs ≈ 3.5–4.5 h).
