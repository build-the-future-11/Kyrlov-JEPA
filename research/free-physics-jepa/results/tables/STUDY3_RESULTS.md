# Study-3 results (amendment 0004)

Run: `confirmatory_s3_v2_20260926T185248Z`
Pretrain source: `confirmatory_s2_v2_20260926T162730Z`
Retrain LMOP: `False`
Git: `1f0430da5c8cc11e24ca94fad8d102eb08a4402f`

## zero_shot

| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) |
|---|------|---------|-----|------|-------------|
| 25 | id | 2.0967 | 0.7778 | 0.6857 | -0.0921 |
| 25 | ood | 2.0759 | 0.8507 | 0.7840 | -0.0667 |
| 100 | id | 3.1950 | 0.7778 | 0.6857 | -0.0921 |
| 100 | ood | 3.1497 | 0.8507 | 0.7840 | -0.0667 |

**Verdict:** `SUPPORTS_HYPOTHESIS`

