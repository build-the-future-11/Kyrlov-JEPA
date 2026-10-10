# Translation-invariant discrete sine coordinates

## Question and implementation

Can the existing classical Dirichlet basis retain its mathematical values when
the same represented rectangular domain is translated to a large origin?
The predecessor constructed physical grid coordinates, subtracted the origin,
then divided by the domain length. At a large origin, floating-point rounding
can erase interior offsets before that subtraction.

The correction constructs the dimensionless interior coordinates directly:

\[
u_i = i/(n+1),\qquad i=1,\ldots,n.
\]

The existing basis therefore evaluates each selected mode as the normalized
outer product of `sin(pi * p * u)` and `sin(pi * q * u)`. The implementation
changes only coordinate construction in `_cached_sine_basis`. Mode selection,
label order, normalization, cached object identity, immutable arrays, solver
choices and returned shapes retain their existing contracts. Domain lengths
still determine the finite-difference operator spacing. The classical basis
itself depends on the interior indices.

This removes an avoidable cancellation, with unchanged asymptotic construction
cost and cache interface. It does not repair physical-coordinate precision in
potential generation, validate learned operators, select a new policy, or
establish improved scientific performance.

## Bounded evidence

The prospective contract and exact incoming canonical state are retained in
`development/sine_coordinates_20261010/`. The contract preceded implementation
and execution. The 20 new cases use independent Kronecker sine matrices, Gram
identities and actual finite-difference kinetic eigenpairs. They exercise
translated and rescaled domains, grids through `n=16`, mode ordering, immutable
cached arrays and both adaptive and fast Ritz paths.

| Retained attempt | Outcome | Interpretation |
| --- | --- | --- |
| `baseline.json` and raw logs | 14 failed, 6 passed | The predecessor fails the declared large-origin coordinate witnesses. |
| `candidate.json` and raw logs | 79 passed | New regressions and the selected inherited contracts pass. |
| `final.json` and raw logs | 85 passed, warnings treated as errors | Six downstream hybrid-spectral cases also pass on the final numerical source. |
| `demo_command.json`, `demo.json` | Completed | Two generated `n=8` zero-potential domains retain identical bases and the same computed energy, `0.38163263408917336`, at origins zero and `1e16`. |

The demo retains its full basis, labels, potential, wavefunction and source
hashes. It is a generated numerical fixture, not a scientific dataset or a
protected evaluation. The evidence includes every command's environment,
elapsed time, child peak RSS, exit status and raw log hashes. Independent source
review checked the algebra, cache behavior, independent oracles and scope without
running additional experiments.

From the repository root, after installing the repository's development
environment, the targeted gate is:

```sh
PYTHONPATH=spectral-krylov-jepa/src:. python -m pytest -q -W error -o addopts= spectral-krylov-jepa/tests/test_sine_translation.py spectral-krylov-jepa/tests/test_ritz_basis_contract.py spectral-krylov-jepa/tests/test_wavefunction_metric_scale.py spectral-krylov-jepa/tests/test_hybrid_spectral.py tests/test_research_evidence_integrity.py
PYTHONPATH=spectral-krylov-jepa/src:. python spectral-krylov-jepa/scripts/demo_sine_translation.py --output /tmp/new-sine-translation.json
```

The demo requires a fresh output path. Original evidence is not overwritten by
reproduction. Executing a future reproduction needs its own recorded compute
allocation; this session's budget closes in `closure.json`.

## Integration and scientific status

This revision follows PR #23 at
`85d8757ce31ec280d50bb54c7f655025b38f076c`. It also incorporates the two independent
root evidence-validator files from main commit
`9d5b456f99512771ca8c3fa41ec99ce8bfb35914`, without changing their bytes. The release
commit records both parents. Existing failures, frozen matrices and hypotheses
remain in the canonical state and its exact predecessor snapshot.

The existing bounded negative/inconclusive disposition remains in force. The
192- and 672-run protected matrices have not been executed. This engineering
correction does not advance a scientific checkpoint or reopen a completed study.
Its draft publication uses `[skip ci]` to avoid executing inherited campaigns
outside this finite local budget. Future scientific adoption needs a source-bound
protocol and its own protected evaluation boundary.
