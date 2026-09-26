# Literature Collision Map — LMOP-JEPA

**Purpose:** Prevent overclaiming. Prior art is established for many nearby ideas.

## Established (NOT our novelty)

| Topic | Status |
|-------|--------|
| Operator learning / FNO / DeepONet | Established |
| Manufactured solutions / MMS for ML | Established |
| Solver-free / physics-informed / PINO / weak-form NO | Established |
| JEPA / latent predictive SSL | Established (vision/audio) |
| Masked PDE / physics SSL | Emerging; not uniquely ours |
| Darcy flow benchmarks | Common |
| Neural operator warm-starts | Established |

## Our experimental question (narrow)

> Latent predictive pretraining on manufactured Darcy triplets may transfer to genuine numerical solution distributions **better than direct manufactured-solution regression** under matched compute and identical low genuine-label budgets.

This is a **comparative transfer hypothesis**, not a claim of inventing manufactured learning or JEPA.

## What would support / falsify

- **Support:** LMOP beats MML on relative \(L_2\) (and preferably energy/flux) at \(N=25\) and/or \(N=100\) with non-overlapping paired bootstrap CIs across seeds; shuffled-physics control worse than correct physics.
- **Falsify:** MML ≥ LMOP; shuffled ≈ correct; spectrum family matters more than objective; neither beats Scratch.

## Language rules

Avoid: “first”, “novel architecture”, “foundation model”, “SOTA”, “breakthrough”.
Prefer: “we evaluate whether…”, “under this protocol…”, “evidence supports / does not support…”.
