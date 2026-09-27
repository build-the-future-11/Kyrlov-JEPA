# Claim gate (auto-generated)

**Verdict:** `NOT_TESTABLE_NO_SEED_VARIANCE`

MML and LMOP predictions do not vary across seeds (single checkpoint per method), so the across-seed CI criterion is degenerate; the frozen gate cannot be applied.

- Matrix complete: True (36/36 rows)
- Shuffled mechanism supported: True
- Git SHA: `1f0430da5c8cc11e24ca94fad8d102eb08a4402f`
- Run: `runs/confirmatory_s3_v2_20260926T185248Z/zero_shot`

## Seed-aggregated relative L2

| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) | LMOP wins (CI) | Seed CI degenerate |
|---|------|---------|-----|------|-------------|----------------|--------------------|
| 25 | id | 2.0967±0.5663 | 0.7778±0.0000 | 0.6857±0.0000 | -0.0921 | no | yes |
| 25 | ood | 2.0759±0.5853 | 0.8507±0.0000 | 0.7840±0.0000 | -0.0667 | no | yes |
| 100 | id | 3.1950±1.3028 | 0.7778±0.0000 | 0.6857±0.0000 | -0.0921 | no | yes |
| 100 | ood | 3.1497±1.3113 | 0.8507±0.0000 | 0.7840±0.0000 | -0.0667 | no | yes |

## Publication language

Under this frozen protocol, evidence **not testable no seed variance** the hypothesis that LMOP-JEPA transfers better than MML-direct at small genuine budgets.

