# Development successor: strict spectral metrics v2

## Question, observation and prospective contract

The retained evaluator gives the zero prediction `residual_rel = 0` and
`residual_true_e = 0`, while its Rayleigh quantities become NaN. Its summary
then drops NaN values independently for each metric. A perfect reported residual
can therefore represent an invalid state, and a nominal cohort size does not
guarantee a shared denominator. These are numerical/evaluation defects, not
evidence of a scientific benefit or failure of Krylov conditioning.

This is an explicit, opt-in **development** successor. This change does not
edit the dependency's evaluator or its callers, frozen protocols, trained
comparisons or retained results. No historical outcome is recalculated here.

Before implementation, the following contract and bounded checks were recorded:

- Inputs are finite real float64-representable values on one declared grid.
  Both reference and predicted states have physical L2 norm within `1e-6` of
  one, using the existing cell-area convention. Zero states are invalid.
- Scale the represented state by its maximum absolute element, then compute
  its Euclidean unit vector. Check physical normalization in log space.
  Residuals are `||H q - E q||_2`, where `||q||_2 = 1`; no denominator floor
  can turn a nonzero residual into a false perfect score.
- Preserve all three energy choices: the supplied energy head, supplied
  reference energy, and Rayleigh quotient. Relative energy errors retain the
  explicit `abs(E - E_true) / (abs(E_true) + 1e-6)` definition. Fidelity and
  sign-aligned distance use the admitted unit vectors; this removes only the
  accepted normalization roundoff, not a materially wrong amplitude.
- Every metric in a summary must exist and be finite for every case. Invalid
  cases remain in a batch receipt and suppress its aggregate. A valid subset
  is not silently substituted for the requested cohort.
- The JSON command admits only `evaluation_mode: development`, records the
  original input bytes and their hash plus implementation/environment identity,
  refuses to overwrite a receipt, and exits nonzero for an invalid audit.
- Generated analytical states, an independent dense finite-difference oracle,
  deliberately invalid states, and actual CLI runs are the verification units.
  No learned model, study seed matrix, protected data or paid compute is used.

Finite arithmetic remains finite arithmetic. Nonfinite operator results and
unrepresentable aggregate statistics are explicit failures. Supplied case IDs,
reference eigenpairs, energy values and a `development` label are declarations;
this interface does not authenticate their source, prove that a reference is a
ground state, authorize protected access, or establish scientific efficacy.

## Concurrent repair and narrowed implementation

During development, existing PR #22 independently supplied the amplitude and
zero-state repair at `0520827ad4d2e0e2063b48b21012bebb167a8485`. This proposal
now stacks on that exact dependency and **reuses its public residual and Rayleigh
functions**. Credit for those equations and 33 associated tests belongs to #22;
the numerical span repair belongs to #21. The initially reproduced zero-state
defect is a historical observation at `be5aea5c658a0fb2da4d0f7cea34560ef62c0c88`,
not an allegation that the new dependency still has that defect.

The distinct remaining defects are executable against #22: a doubled physically
normalized prediction receives fidelity 4 and infidelity -3, and a summary with
one NaN still labels the cohort `n = 2` while computing that metric from one
value. This version admits normalized states explicitly and makes complete-case
accounting mandatory. It also supplies a usable audit API/CLI that retains invalid
cases and prevents an existing report from being overwritten.

Independent engineering review found two problems in the initial new candidate:
centering variance on an already rounded mean mismeasured the spread of adjacent
floats, and inferring NumPy dtype from mixed lists could erase a Boolean or a
large integer before validation. The final implementation validates each original
list scalar first and computes the mean and variance of represented inputs with
exact rational arithmetic. It takes a scaled float square root only at the end.
Nonzero mean or standard deviation rounding to zero is refused. This is explicit
finite arithmetic handling, not arbitrary-precision recovery of measurements.

## Executable interface

The Python functions are `evaluate_example`, `summarize_metrics`, and
`audit_examples` in `spectral_krylov_jepa.evaluation.metrics_v2`. Case records
contain exactly `case_id`, `potential`, `psi_true`, `e_true`, `psi_hat`, and
`e_hat`. Identifiers must be unique nonempty strings. All field arrays have shape
`(n_interior, n_interior)`. The grid declares the integer interior size and all
four coordinate limits; no field layout or unit spacing is inferred.

For the retained development fixtures, run from `spectral-krylov-jepa/`:

```sh
PYTHONPATH=src python -m spectral_krylov_jepa.evaluation.metrics_v2 \
  research/verification/strict_metrics_v2_20261010/valid_input.json \
  /tmp/krylov_valid_fresh_receipt.json
PYTHONPATH=src python -m spectral_krylov_jepa.evaluation.metrics_v2 \
  research/verification/strict_metrics_v2_20261010/invalid_input.json \
  /tmp/krylov_invalid_fresh_receipt.json
```

Choose fresh output paths. A valid audit exits 0. An invalid audit or an existing
receipt exits 2; an invalid input still leaves a report with its original byte
snapshot, reason and SHA-256. No invalid case contributes to a partial aggregate.
When all cases have valid metrics but a cohort statistic is unrepresentable,
the report retains those metrics and marks the overall audit invalid with a
`summary_error`. These receipts record implementation hashes and dependency
versions; bitwise replay across different numerical libraries is not promised.

## Completion record

**134 tests passed, zero failures/skips:** 53 new cases and 81 inherited
numerical/physics cases, including the dependency's amplitude regressions. The
actual CLI produced a valid two-case summary (exit 0), and retained all three
cases while withholding the summary when a zero prediction was added (exit 2).
Fatal-error/import lint and syntax checks pass. Exact commands, source/output
hashes, dependency versions, historical witnesses and raw output are retained in
`research/verification/strict_metrics_v2_20261010/receipt.json`.

An independent agent reviewed source and tests, reproduced the two initial
candidate defects, and then passed all 53 final focused tests plus 11 independent
arithmetic/admission witnesses after correction. This is engineering review, not
human or scientific peer review. The publishing commit uses `[skip ci]` because
inherited broad spectral-path workflows launch research execution. No hosted
CI success or full scientific-suite execution is claimed.

The methods/reproducibility and bounded negative/inconclusive disposition
remains authoritative. A future study must declare this evaluator version and
its admitted reference/split provenance; these checks establish no learning
advantage or new scientific checkpoint.
