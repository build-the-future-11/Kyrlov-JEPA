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
