# LMOP-JEPA / Free-Physics JEPA — paper draft skeleton
# After overnight run, paste `RESULTS_AUTO.md` into §Results.

## Title

Does latent predictive pretraining on manufactured Darcy examples transfer better than direct manufactured-solution regression under small genuine-label budgets?

## Abstract (fill after claim gate)

Under a frozen protocol comparing **MML-direct** vs **LMOP-JEPA** on 64×64 Darcy flow with genuine budgets \(N\in\{25,100\}\) and seeds \(\{11,23,47\}\), we find: **[AUTO: see `paper/RESULTS_AUTO.md`]**.

## 1. Question

We ask a narrow comparative transfer question (not an architecture-novelty claim): whether joint-embedding latent pretraining on solver-free manufactured Darcy triplets improves few-shot genuine-solution accuracy relative to direct manufactured-solution regression, under matched manufactured corpora, matched FNO backbones, and identical genuine splits.

## 2. Protocol (frozen)

- PDE: \(-\nabla\cdot(a\nabla u)=f\) on \([0,1]^2\), Dirichlet zero; cell-centered FD with harmonic face averages.
- Grid: confirmatory \(n=64\).
- Manufactured primary family: MIXED frequency.
- Genuine ID: log-GRF permeability \(\ell=0.20\); primary OOD: \(\ell=0.08\).
- Methods: Scratch, MML-direct, LMOP-JEPA (amendment 0002 variance hinge).
- Budgets / seeds: \(N\in\{25,100\}\), seeds \(\{11,23,47\}\).
- Primary metric: relative \(L_2\); secondary: \(H^1\), energy, flux, PDE residual.
- Mechanism control: shuffled-physics LMOP pretraining.

See `protocol.yaml`, `RESEARCH_PROTOCOL.md`, `amendments/`.

## 3. Methods

Brief descriptions of Scratch / MML / LMOP; shared `FNO2d` trunk; EMA target JEPA with mask ratio 0.4; VICReg-style variance hinge (amendment 0002).

## 4. Results

**Replace this section with the contents of `RESULTS_AUTO.md` after the overnight run.**

Also include:
- `figures/confirmatory_label_efficiency.png`
- `figures/confirmatory_synthetic_real_norms.png` (diagnostic mismatch only)

## 5. Limitations

- Synthetic↔real solution-norm scale gap (documented; not used mid-study redesign).
- Amendment 0002 changed LMOP objective after an aborted confirmatory pretrain; MML checkpoint reused.
- Laptop/MPS compute; CJSJ core only (no \(N=250\) / LOW ablation unless run).
- Smoke metrics are not scientific evidence.

## 6. Disallowed claims

Do not claim novelty of manufactured solutions, operator learning, PINO/weak-form methods, or generic physics JEPA. Do not cite Spectral Krylov-JEPA as LMOP evidence.

## Reproducibility

```bash
./research/free-physics-jepa/scripts/overnight_confirmatory.sh
```

Manifest SHA and git SHA are recorded in `runs/<id>/summary.json` and `results/tables/claim_gate.json`.
