# Protocol Amendment 0002 — Anti-collapse variance hinge

**Date:** 2026-09-26  
**Status:** ACTIVE  

## Problem

Confirmatory LMOP pretraining aborted with `latent_var=7.4e-7` under the batch-channel variance diagnostic after LayerNorm. Smoke (smaller model/steps) did not trip this gate. Pure EMA+LN JEPA can numerically look “collapsed” on this particular scalar even while loss decreases.

## Amendment

1. Add optional VICReg-style **variance hinge** on predictor latents:  
   \(\mathcal L += \lambda_v \sum_c \mathrm{ReLU}(\gamma - \sqrt{\mathrm{Var}_B[z_c]+\epsilon})\) with \(\lambda_v=1\), \(\gamma=1\) (frozen a priori).
2. Collapse monitor uses **mean per-channel batch std of spatially pooled latents**; fail only if hinge-corrected run still has mean std \(<10^{-4}\) after 200 steps.
3. Does **not** change PDE, splits, metrics, OOD, or label budgets.

## Invalidates

Smoke LMOP checkpoints remain smoke-only. Confirmatory MML checkpoint from the failed run may be reused; LMOP must be retrained with this amendment.
