# Study-2 publish checklist (amendment 0003)

## Done

- [x] Study-1 archived as negative/null (amendments 0001–0002)
- [x] Amendment 0003 implemented (matched MMS, norm, same-encoder block-JEPA, fixed shuffle, freeze FT)
- [x] Smoke v2 gate passed
- [x] Confirmatory Study-2 complete: **36/36** rows (`confirmatory_v2.csv`)
- [x] Norm ratio mfg/genuine = **0.978** (fixed)
- [x] Shuffle mechanism supported = **True** (15.83 > 14.58)
- [x] Claim gate written (`claim_gate_v2.md`)
- [x] Figures: `confirmatory_v2_label_efficiency.png`, `confirmatory_v2_synthetic_real_norms.png`
- [x] `paper/RESULTS_AUTO.md` regenerated
- [x] Results pasted into `paper/DRAFT.md`
- [x] Study-2 result note committed

## Verdict (frozen)

**`FALSIFIES_HYPOTHESIS`**

MML beats LMOP on relative \(L_2\) in every cell (ID/OOD × N=25/100). No LMOP CI win. Hypothesis that LMOP transfers better than MML is **not supported**.

## Remaining (author only — optional polish)

- [ ] Abstract/intro prose polish in `paper/DRAFT.md`
- [ ] Choose venue / format (arXiv note vs workshop)
- [ ] Optional: plot Study-1 vs Study-2 side-by-side as ablation narrative
- [ ] Do **not** run more seeds hoping for a win without a new amendment

## Do not

- Cite smoke as evidence
- Cite Krylov-JEPA as LMOP support
- Claim novelty of MMS / FNO / JEPA
