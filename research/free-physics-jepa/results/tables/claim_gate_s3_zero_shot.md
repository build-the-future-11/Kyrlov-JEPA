# Claim gate (auto-generated)

**Verdict:** `SUPPORTS_HYPOTHESIS`

LMOP beats MML with non-overlapping seed CIs on ['N=25/id', 'N=25/ood', 'N=100/id', 'N=100/ood']; shuffled-physics control worse than correct.

- Matrix complete: True (36/36 rows)
- Shuffled mechanism supported: True
- Git SHA: `1f0430da5c8cc11e24ca94fad8d102eb08a4402f`
- Run: `/Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s3_v2_20260926T185248Z/zero_shot`

## Seed-aggregated relative L2

| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) | LMOP wins (CI) |
|---|------|---------|-----|------|-------------|----------------|
| 25 | id | 2.0967±0.5663 | 0.7778±0.0000 | 0.6857±0.0000 | -0.0921 | yes |
| 25 | ood | 2.0759±0.5853 | 0.8507±0.0000 | 0.7840±0.0000 | -0.0667 | yes |
| 100 | id | 3.1950±1.3028 | 0.7778±0.0000 | 0.6857±0.0000 | -0.0921 | yes |
| 100 | ood | 3.1497±1.3113 | 0.8507±0.0000 | 0.7840±0.0000 | -0.0667 | yes |

## Publication language

Under this frozen protocol, evidence **supports hypothesis** the hypothesis that LMOP-JEPA transfers better than MML-direct at small genuine budgets.

