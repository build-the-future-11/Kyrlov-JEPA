# Development revision: amplitude-invariant wavefunction metrics

This prospective numerical repair concerns the spectral Krylov evaluator. It
does not revise retained results, reopen a frozen study, or change the separate
LMOP and CJSJ subprojects. The original learning advantage remains unestablished.

## Observation and discriminating check

At parent commit `be5aea5c658a0fb2da4d0f7cea34560ef62c0c88` (PR #21),
the residual divided by an absolute denominator floor, and the Rayleigh quotient
used unscaled squared amplitudes. Tiny but nonzero predictions could produce a
false near-zero residual or undefined energy; large amplitudes could overflow.
A zero wavefunction was assigned a zero residual despite being undefined.

Both quantities are homogeneous of degree zero in the wavefunction. The
prediction is that changing its nonzero amplitude preserves the result. On a
2-by-2 interior unit-square grid with zero potential, the first coordinate
vector has Rayleigh energy 18 and residual `sqrt(41.5)` at energy 17. This gives
a closed-form oracle independent of any trained model or eigensolver.
The 33 new tests produced 23 failures and 10 passes against the parent source;
the original failure output is retained.

## Change and resulting contract

The evaluator removes arbitrary state amplitude before applying the unchanged
Hamiltonian. Rayleigh dot products and residual norms are also scaled before
their reductions. Tests cover nonzero scales from `1e-300` to `1e300`, signs,
an exact constant box eigenmode, potential/energy shifts, and representable
extreme energy values. Flat and grid-shaped states retain equivalent behavior.

Zero, nonfinite, complex, and wrong-size states now raise `ValueError`; residual
energies must be finite. Unrepresentable operator actions or final metrics also
raise explicitly. This is a compatibility change for future evaluation runs,
not a replacement of any frozen output. Fidelity and sign-aligned error keep
their existing normalized-state contracts and are outside this repair.

## Verification and limits

All 33 new tests and four existing metric tests pass (37 total). The new tests
use analytic fixtures; the existing tests use tiny free-box solver fixtures.
These are implementation checks, not an empirical study. Focused lint, syntax
compilation, and whitespace checks pass. Exact commands, versions, file hashes,
and raw output are in
`research/verification/wavefunction_metric_scale_20261010/receipt.json`.

The implementation remains real float64 and cannot recover information already
lost in an input array. Representable results requiring an unrepresentable
Hamiltonian action are rejected rather than estimated. No training, protected
outcomes, scientific campaign, dataset access, or paid compute was used; no
independent reviewer is claimed for this revision.

Next action: review the explicit invalid-state error behavior and integrate
after PR #21. Any new study must identify this version and follow its own
scientific execution gate; these tests establish no learning advantage.
