# Confirmatory benchmark protocol freeze — 29 September 2026

**Status:** FROZEN BEFORE CONFIRMATORY OUTCOMES.  
**Parent source tree:** `6a0928883dce3f8b02513f8c78f0e7907e8c5b21`.

This package does not modify the previously frozen `paper/experiment_protocol.md` or
`paper/CONTROL_PROTOCOL_2026-09-27.md`. It defines a separate bounded benchmark
layer around the already-existing fresh calibration runner.

The neural methods are not reimplemented here. `scratch_physics`, `krylov_physics`
and `krylov_projected_physics` are taken from
`scripts/19_run_confirmatory_calibration.py`, which already fixes the fresh corpus,
seeds, label budgets, pretraining steps, downstream objective and OOD evaluation.
The classical methods added here are complete implementations with no outcome-based
selection:

- `fixed_ritz_3x3`: fixed 9-vector sine Rayleigh-Ritz.
- `pt2_3x3`: second-order perturbative energy with first-order state in the same
  fixed 9-vector sine basis.
- `pt2_to_krylov_ritz_d3`: PT2/first-order perturbative state as the deterministic
  start vector, then depth-3 fully reorthogonalized Lanczos and Rayleigh-Ritz.

## Frozen selection rules

1. No basis side, perturbation order, Krylov depth, seed, split or label budget may
   be changed after confirmatory test outcomes are inspected.
2. Neural checkpoints are selected by validation only, exactly as in the existing
   `finetune()` path. Test data never selects checkpoints.
3. Every declared method is retained in output. Failed/missing cells remain explicit.
4. Neural ID results are reported at all frozen label budgets. OOD neural results
   are reported at the maximum label budget, matching runner 19. Label-free classical
   methods are evaluated on every frozen test split.
5. Compute is reported in separate strata: offline optimizer work, reusable setup,
   and online query work. Wall-clock time is descriptive, not a claim of FLOP parity.
6. Smoke outputs are engineering validation only and cannot be used as efficacy evidence.

No confirmatory run is started by default. The runner requires the explicit
`--execute-confirmatory` flag to launch the pre-existing neural training/evaluation
pipeline. `--smoke` runs only a tiny deterministic synthetic infrastructure test.
