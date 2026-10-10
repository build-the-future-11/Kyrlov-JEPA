# Development repair: ritz numerical span

Date: 2026-10-10. Base: `d61c26ba40fa0a6e8ee0bc39287c17faa7367bef`. Evidence class: engineering regression verification.

## Failure and scientific scope

Reduced QR completed dependent columns with arbitrary extra directions. An unchecked orthonormal fast path multiplied energies when its basis was scaled. One-pass adaptive orthogonalization retained large low-mode cancellation error.

No original trained comparison matrix, protected split, or future confirmatory outcome is read or run. Existing protocols and submission packages remain frozen.

## Implementation contract

Use a scale-normalized SVD to retain numerical rank, validate the fast-path Gram matrix, and reorthogonalize adaptive proposals twice against fixed and accepted directions.

Dependent, zero-padded, rescaled and permuted representations of a span yield the same rank and Ritz result; invalid fast-path assumptions fail explicitly.

## Verification and retained failures

The exact reproduction command, environment versions, source SHA-256 identities, test output, and exit status are in `spectral-krylov-jepa/research/verification/ritz_numerical_span_20261010/receipt.json`. The canonical state retains the base-source regression failure counts and observed review failures. This record was written after exploratory defect discovery, and is not a preregistered confirmatory experiment.

An independent agent examined the modified source and regression assertions. This is project-controlled review, not external scientific replication.

## Limits and next action

Rank is a float64 numerical rank at eps*max(shape)*largest singular value. The 1e-10 absolute and 1e-8 relative Gram tolerances are declared numerical settings, not claims of exact symbolic rank.

Methods/reproducibility package complete; efficacy not established. Original retained package is bounded negative/inconclusive.

- Review the numerical behavior change separately from the frozen study; any new result must name this code revision.
- Respect the existing scientific execution gate and publish no efficacy claim without the required matched evidence.
- Preserve separate LMOP and CJSJ subproject state; this state applies to spectral-krylov-jepa only.


## Follow-up: wavefunction amplitude normalization

The previous normalization rejected fields scaled by `1e-250` and returned an
all-zero array for otherwise valid fields scaled by `1e250`. Their directions
are unchanged mathematically. The normalizer now keeps the exact ordinary
arithmetic path and uses max-scaled normalization when the squared norm cannot
be computed reliably. It rejects zero, nonfinite, complex and wrong-size fields
explicitly. The real-valued Hamiltonian contract is unchanged.

Thirteen new analytic fixtures include positive/negative amplitude scaling,
input immutability, weighted norm one, exact ordinary arithmetic, invalid inputs
and a complete spectral coefficient reconstruction. They produced 9 failures /
4 passes on the preceding normalizer and all pass after repair. Together with
existing eigensolver, metric, hybrid-spectral and Ritz-contract tests the focused
suite is **47 passed** with warnings treated as errors. The recorded old Ritz
verification receipts remain unchanged and keep their original scope.

No frozen experiment, result table, training matrix or submitted artifact was
modified or replayed. This is a prospective numerical correction, with no
change to the scientific efficacy conclusion.
