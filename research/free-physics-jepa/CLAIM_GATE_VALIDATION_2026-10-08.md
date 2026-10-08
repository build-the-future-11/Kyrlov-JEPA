# Frozen-matrix claim-gate validation

Source reviewed: `d61c26ba40fa0a6e8ee0bc39287c17faa7367bef`.

## Defect and repair

The LMOP-JEPA aggregation gate treated a row count at least as large as the
expected matrix as completeness. It did not require the declared seed identities
or one unique measurement for each method, seed, label budget, and distribution.
It also converted the shuffled-control flag with Python truthiness.

Three synthetic counterexamples returned `SUPPORTS_HYPOTHESIS` at that source:

| Counterexample | Earlier behavior | Repaired behavior |
| --- | --- | --- |
| All seeds replaced by undeclared seed identities | Complete, supports | `INVALID_MATRIX` |
| One scratch cell omitted and another duplicated, preserving 36 rows | Complete, supports | `INVALID_MATRIX` |
| Mechanism flag set to JSON string `"false"` | Mechanism supported | `INCONCLUSIVE_MECHANISM` |

The gate now validates the exact Cartesian product of the existing frozen axes,
rejects duplicated or unexpected cells, rejects coerced/fractional identities,
and requires every primary relative-L2 value to be finite and nonnegative.
Missing cells yield `INCOMPLETE_MATRIX`; malformed cells yield `INVALID_MATRIX`.
Neither case produces support cells or scientific comparisons. The report records
missing identities and validation errors. Shuffled-control evidence requires a
JSON boolean and records malformed-artifact diagnostics.

The statistical decision rule, seed list, label budgets, split manifests, model
code, adaptation protocols, and retained results are unchanged. In particular,
mixed adaptation rows must be separated into their existing per-adaptation
tables before being evaluated by this gate.

## Validation

`python -m pytest -q tests/test_claim_gate.py`: **34 passed** in 0.45 seconds using
the existing research Python environment. The original four tests still pass;
the additional cases cover matrix identity and coverage, non-finite metrics,
malformed control flags, valid row ordering, and malformed protocol axes.
The edited files also pass Python compilation.

Five retained tables, containing 180 rows in total, were passed through both the
original and repaired aggregation logic with the same stored shuffled-control
artifact. Outputs were written only to temporary directories. All verdicts,
matrix-completeness values, and support-cell lists were identical:

| Retained table | Verdict before and after |
| --- | --- |
| `confirmatory.csv` | `FALSIFIES_HYPOTHESIS` |
| `confirmatory_v2.csv` | `FALSIFIES_HYPOTHESIS` |
| `study3_confirmatory_zero_shot.csv` | `NOT_TESTABLE_NO_SEED_VARIANCE` |
| `study3_confirmatory_probe.csv` | `FALSIFIES_HYPOTHESIS` |
| `study3_confirmatory_low_lr_ft.csv` | `FALSIFIES_HYPOTHESIS` |

This verifies aggregation behavior and preservation of retained conclusions.
It is not a model rerun or independent scientific replication. No training,
protected evaluation, new benchmark, or experiment-selection decision was made.
The gate validates table structure and required values; it does not independently
prove that upstream data collection or provenance declarations are truthful.
