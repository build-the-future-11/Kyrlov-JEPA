# Development ledger input verification — 7 October 2026

The generic policy-freeze CLI previously coerced fractional counts with `int(...)`, stringified invalid case IDs, and ignored explicit input declarations that protected outcomes or exact query solves had been opened. It then emitted `protected_outcomes_opened: false`. It also recorded only a mutable ledger path.

The loader now preserves input types and requires explicit JSON `false` values for both `protected_outcomes_opened` and `query_exact_eigensolves_performed`. The policy routine rejects fractional, string, boolean, nonpositive, and inconsistent operator counts, and rejects invalid residuals/case IDs. The CLI parses and hashes the same input bytes and records their SHA-256. It refuses to overwrite an existing frozen policy or its input ledger.

The rank rule, quantiles, residual threshold computation, candidate set, budget, physics solver, frozen 5 October policy, and existing result/manuscript bytes are unchanged. This addendum documents input verification; it does not amend the earlier frozen scientific protocol or authorize a protected comparison.

## Offline verification

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover \
  -s spectral-krylov-jepa/tests -p test_recycled_ledger_integrity.py -v
```

Ten tests pass, including counterexamples for explicit protected access, silently truncated ranks, invalid labels/residuals, duplicate/rank-dependent cases, exact input-byte hashing, and a CLI attempt to overwrite an existing policy. Fixtures are deliberately synthetic development ledgers; they are not new experiment outcomes.

When using `scripts/18_freeze_recycled_ritz_policy.py`, include both required boundary declarations in the input ledger and choose a fresh output path. Older minimal ledgers without those declarations now fail closed and must be reconciled with their authentic source before use. A declaration and a digest identify claimed provenance; they do not prove that an external producer respected the boundary.

The current methods and bounded negative/inconclusive claim remains unchanged. No full efficacy comparison or protected outcome was run for this change.
