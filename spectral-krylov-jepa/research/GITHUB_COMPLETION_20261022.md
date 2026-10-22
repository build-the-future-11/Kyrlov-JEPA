# Numerical stack completion — 22 October 2026

This prospective engineering revision preserves the complete PR21–24 stack and
all of its source-bound records. Numerical rank, two-pass reorthogonalization,
amplitude-invariant metrics, strict complete-cohort evaluation and direct-index
Dirichlet sine coordinates retain their reviewed behavior. The original frozen
studies, negative/inconclusive findings and unestablished learning advantage are
unchanged. No scientific result has been recomputed or upgraded.

## Previously recorded normalization limit

The preceding integration explicitly retained an unexecuted analytic witness:
individual component squares could be subnormal even though multiplication by
a large grid cell area made the final norm finite. For four equal field values
`1.5 * 2^-537` and cell area `2^1000`, the returned weighted norm squared was
approximately `9/8`. The final-norm check alone did not detect the earlier loss.

A new bounded session specified this correction before execution, without
reopening prior budgets. Ten tests use a 200-digit Decimal oracle over the exact
represented input floats and area. They cover signed nonuniform fields, small,
ordinary and large cell areas, and very small/large global amplitudes. The exact
old source failed two cases and passed eight controls; its output is retained.

The normalizer now uses its existing scaled branch whenever any nonzero
component square is subnormal or rounds to zero. Safe ordinary squares retain
the original operation order and exact historical arithmetic. The finite-area,
real/nonzero/finite field and finite-result admission rules are unchanged.

## Verification and remaining gate

The complete selected stack passed **183 tests, zero failures or skips**, with
warnings treated as errors in 3.17 seconds. This includes all 173 previously
selected physics, numerical, cohort, sine and evidence-integrity checks plus
the ten independent Decimal cases. Critical lint passed on all changed runtime
modules and the new test file. Two test commands used 4.05 seconds in total;
there was one numerical repair cycle. No training, scientific campaign,
protected outcome access or paid compute occurred.

The new contract, original failure output, final output, source identities and
closed local receipt are in `development/github_completion_20261022/`. Earlier
receipts and limitations retain their original identities and scope. This
record establishes the scoped numerical repair, not universal accuracy for
every unrepresentable input or any empirical learning advantage.

Before merging, the parent must independently review the exact final tree and
read a successful engineering CI run for that published head. The separately
reviewed workflow change preserves automatic engineering tests and requires
explicit manual dispatch for scientific campaigns. No campaign is dispatched
as part of this completion, and no hosted success is claimed here.
