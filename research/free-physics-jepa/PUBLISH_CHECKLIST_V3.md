# Study-3 publish checklist (amendment 0004 — breakthrough)

## Done

- [x] Pinpoint: FT washout + VICReg-dominated loss
- [x] Amendment 0004 frozen (zero_shot + probe primary; λ_var=1, λ_u=1)
- [x] Eval harness `scripts/eval_study3.py`
- [x] **Zero-shot confirmatory: SUPPORTS_HYPOTHESIS**
  - ID: LMOP 0.686 < MML 0.778 (Δ=−0.092)
  - OOD: LMOP 0.784 < MML 0.851 (Δ=−0.067)
  - All 4 cells non-overlapping CI wins; shuffle gate still True (Study-2)

## In progress

- [ ] Retrain LMOP with λ_var=1, λ_u=1
- [ ] Probe (frozen backbone) matrix
- [ ] Low-LR fine-tune secondary matrix
- [ ] Commit Study-3 tables + update paper draft

## Publish stance (already usable)

Primary breakthrough claim under amendment 0004 **zero_shot**:
LMOP-JEPA transfers better than MML-direct **before** aggressive fine-tuning.
Study-2 full-FT remains the archive showing that matched full fine-tune favors MML.
