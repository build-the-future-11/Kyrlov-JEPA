# LMOP-JEPA / Free-Physics JEPA — paper draft

## Title

Does latent predictive pretraining on manufactured Darcy examples transfer better than direct manufactured-solution regression under small genuine-label budgets?

## Abstract

We evaluate a narrow comparative transfer hypothesis on 64×64 Darcy flow: whether LMOP-JEPA (joint-embedding latent pretraining on solver-free manufactured solutions) improves few-shot genuine-solution accuracy relative to MML-direct (manufactured regression) under matched corpora, matched FNO backbones, and identical budgets \(N\in\{25,100\}\) with seeds \(\{11,23,47\}\).

**Study-1** (mismatched MMS scale, weak JEPA transfer, broken shuffle) was a null/negative pipeline result. **Study-2** (amendment 0003: matched \(\|u\|_2\), input normalization, same-encoder multi-block JEPA + VICReg, fixed `u_vs_af` shuffle, freeze-then-unfreeze fine-tune) repaired those failure modes: manufactured/genuine norm ratio ≈0.98 and shuffled pretrain loss exceeded correct (mechanism gate passed). Under the frozen claim gate, evidence still **falsifies** the hypothesis that LMOP beats MML: MML attains lower relative \(L_2\) in every ID/OOD × budget cell, with no non-overlapping seed-CI win for LMOP.

## 1. Question

Latent predictive pretraining on manufactured Darcy triplets may transfer to genuine numerical solutions better than direct manufactured-solution regression under small genuine-label budgets. This is a comparative transfer hypothesis, not an architecture-novelty claim.

## 2. Protocol

Frozen in `protocol.yaml` + amendments 0001–0003. Decisive comparison: **MML-direct vs LMOP-JEPA**; Scratch secondary. Primary metric: relative \(L_2\). Primary OOD: permeability correlation-length shift.

## 3. Methods

Scratch; MML-direct; LMOP-JEPA v2 (same 2-channel FNO encoder as downstream, multi-block mask, VICReg, hybrid \(λ_u=0.1\), full backbone transfer). Fine-tune: freeze backbone 10 epochs then unfreeze.

## 4. Results (Study-2 confirmatory)

**Verdict: `FALSIFIES_HYPOTHESIS`**

Run: `confirmatory_s2_v2_20260926T162730Z` · Matrix: 36/36 · Shuffle supported: True · Norm ratio: 0.978

| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) | LMOP wins (CI) |
|---|------|---------|-----|------|-------------|----------------|
| 25 | id | 0.519±0.006 | **0.503±0.001** | 0.529±0.006 | +0.026 | no |
| 25 | ood | 0.540±0.004 | **0.532±0.001** | 0.538±0.003 | +0.006 | no |
| 100 | id | 0.503±0.006 | **0.500±0.002** | 0.518±0.002 | +0.018 | no |
| 100 | ood | 0.538±0.002 | **0.533±0.002** | 0.544±0.007 | +0.011 | no |

Shuffled LMOP loss 15.83 > correct 14.58 (mechanism gate passes; still no transfer win).

Figures: `figures/confirmatory_v2_label_efficiency.png`, `figures/confirmatory_v2_synthetic_real_norms.png`.

## 5. Interpretation

Matching MMS scale and fixing JEPA/shuffle pathologies was necessary but not sufficient for an LMOP win. On this Darcy few-shot protocol, **direct manufactured regression (MML) is the stronger transfer objective**; latent JEPA pretraining does not improve relative \(L_2\) and is often slightly worse than Scratch after fine-tuning.

## 6. Limitations

Laptop/MPS compute; CJSJ core only (\(N\in\{25,100\}\)); hybrid \(λ_u\) couples JEPA to regression; Study-2 amended mid-program after Study-1 autopsy (documented a priori in amendment 0003 before Study-2 outcomes were used for claims).

## 7. Disallowed claims

No novelty claims for manufactured solutions, operator learning, or JEPA. Krylov-JEPA is unrelated. Smoke metrics are not evidence.

## Reproducibility

```bash
./research/free-physics-jepa/scripts/overnight_confirmatory_v2.sh
```

Artifacts: `results/tables/confirmatory_v2.csv`, `results/tables/claim_gate_v2.json`.
