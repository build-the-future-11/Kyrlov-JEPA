# Study-2 confirmatory report (amendment 0003)

**Status:** COMPLETE  
**Run:** `confirmatory_s2_v2_20260926T162730Z`  
**Git SHA:** `815c31011cf8221486c76f6d1352f9e79cec172e`  
**Claim gate:** `FALSIFIES_HYPOTHESIS`

## Pipeline repairs (vs Study-1)

| Check | Study-1 | Study-2 |
|--|--|--|
| Norm ratio mfg/genuine | ~7.4 | **0.978** |
| Shuffle mechanism gate | Fail (shuffle easier) | **Pass** (15.83 > 14.58) |
| Matrix | 36/36 | 36/36 |

## Decisive relative \(L_2\) (seed mean ± std)

MML wins every cell. LMOP is worse than MML by ~0.006–0.026 and often worse than Scratch on ID.

See `results/tables/claim_gate_v2.md` and `paper/DRAFT.md`.

## Publish stance

Honest negative confirmatory result under a repaired protocol: latent JEPA pretraining on matched manufactured Darcy does **not** beat direct manufactured regression at \(N\in\{25,100\}\).
