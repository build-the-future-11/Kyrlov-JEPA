# Recycled Ritz control — development freeze protocol

This document closes the policy-selection ambiguity in issue #13 without opening any
confirmatory outcome. It governs the classical recycled-subspace control only.

## Frozen information boundary

Policy selection may consume only records explicitly marked `source_role=development`.
Each candidate rank must be evaluated on the identical set of development case IDs.
The policy-freeze code rejects confirmatory/protected roles, missing ranks, duplicate case
IDs, rank-dependent case sets, non-finite residuals, and inconsistent operator ledgers.

The selection statistic is the normalized full-space Ritz residual

[
|H x - 	heta x|_2 / max(|H x|_2, epsilon).
]

No protected exact energy, protected eigenvector, protected success label, or held-out
method comparison is an input to policy selection.

## Pre-outcome rule

Before any protected comparison, freeze:

- candidate ranks;
- maximum allowed `H_new @ U` operator applications;
- ranking quantile (default 0.90);
- threshold quantile (default 0.95);
- near-best factor (default 1.10);
- exact development ledger identity.

For every budget-legal candidate rank, compute the conservative empirical ranking
quantile using NumPy's `method="higher"`. Let the smallest ranking residual be
`r_best`. The selected rank is the **smallest** rank whose ranking residual is at most
`1.10 * r_best` unless a different factor was explicitly frozen beforehand.

The refresh threshold is then the conservative threshold quantile of the selected rank's
development residuals. No test-set adjustment is permitted.

This rule deliberately trades a small amount of development residual quality for lower
operator cost when the cheaper rank is already near the best observed development
residual profile.

## Test-time accounting

For a rank-`k` query, the core control must recompute `H_new @ U` and records `k`
full operator applications before any fallback. The reduced solve is reported separately.
If the frozen residual threshold fires, any refresh or fallback solver work must be added
to the same test-time budget charged to competing methods; it is not hidden inside the
recycled-control API.

## Reproduction

Freeze a policy with:

```bash
python scripts/18_freeze_recycled_ritz_policy.py \
  --development-ledger path/to/development_residuals.json \
  --output path/to/recycled_ritz_policy.json \
  --candidate-ranks 1 2 4 8 \
  --max-operator-applications 8 \
  --ranking-quantile 0.90 \
  --threshold-quantile 0.95 \
  --near-best-factor 1.10
```

The input ledger must be a JSON object with `source_role: "development"` and a
`records` list. Each record must provide `case_id`, `rank`,
`residual_relative_to_hx`, and `operator_applications`.

The generated policy artifact explicitly records that protected outcomes were not opened.

## What this does not authorize

This protocol does not run the protected confirmatory study, does not choose a policy from
protected performance, and does not establish that recycled Ritz or Krylov-JEPA is
superior. The result-bearing comparison remains fail-closed until the development ledger,
candidate-rank set, budget, and resulting policy artifact are frozen and reviewed.
