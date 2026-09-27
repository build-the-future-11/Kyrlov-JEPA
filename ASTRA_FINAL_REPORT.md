# ASTRA final report — Spectral Krylov-JEPA

**Date:** 27 September 2026. **Scope:** user megaprompt 21, applied to
`/Volumes/PRO-BLADE/Kyrlov-JEPA/spectral-krylov-jepa`.

**Verdict:** local engineering and bounded control execution pass. The retained
scientific result is a negative baseline audit. Krylov transfer improvement is
unestablished. The full experiments, independent reproduction and human submission
requirements remain open. **Not release-ready, paper-ready or submission-ready.**

## Starting state

Starting commit: `22e202d` (full SHA in `astra/2026-09-27/start_head.txt`), branch
`master`; only `.serena/` was untracked. The repository contained a validated
finite-difference Hamiltonian/Lanczos implementation, four neural methods, a
frozen protocol, a historical one-seed development run, and an outline rather
than a completed Krylov manuscript. A neighboring LMOP project was already
separately audited and was preserved.

The retained improved smoke manifest actually contains 120 unlabeled potentials
and 88 labeled examples: train 40, validation 12, test-ID 12, and three OOD groups
of eight. This is distinct from the smaller smoke layout in the frozen planning
document. The decisive experiment had never run; no existing result was relabeled
as that experiment.

## Changes made

- Added an executable fourteen-variant supplement spanning context order,
  update frequency, subspace construction, normalization, coefficient loss,
  pretraining budget and representation width. Added a normalized-powers control
  with the same neural architecture and shapes as default K2.
- Added fixed free-box, sine-subspace Ritz, training-mean and linear ridge baselines
  and an audit with retained per-example metrics, paired potential bootstrap
  intervals and generated figures/tables.
- Made the main runner use run-local manifests, source/configuration identities,
  data hashes and an exclusive OS lock; incompatible/legacy resumes fail closed.
  Tests now keep their manifests local. Prevented an empty pretraining loader from
  entering an endless loop. Default scientific training behavior is preserved.
- Added width propagation through pretraining, fine-tuning and evaluation,
  target-update frequency, final-latent normalization control, loss histories,
  parameter counts and whole-process peak memory reporting.
- Documented the distinction between corpus depth and context order, classical
  Krylov facts and neural hypotheses, one-SSL-seed versus downstream-seed variation,
  and the remove-V transfer-path confound.
- Wrote the full technical report with three scoped proofs, a CJSJ-template Word
  draft, separate editable PowerPoint figure, AI prompt disclosure, literature
  verification, claim/limitation/release ledgers, deterministic reproduction
  commands, local verification script and an unexecuted hosted CI definition.

## Work actually executed

| Check or experiment | Observed outcome | Receipt |
|---|---|---|
| Initial workspace tests | 36 passed | tool execution; starting snapshot |
| Final regression suite | 39 passed | `astra/2026-09-27/tests.log` |
| Packaged local verification | PASS: all 39 tests, lint, compilation and physics | `local_verification.log` |
| Critical Python lint | PASS (`E9,F63,F7,F82`) | `lint.log` |
| Python compilation | PASS | local compileall, final verification |
| Physics validation, grid 32 | PASS; eigenpair residual 1.519e-08; Lanczos recurrence 2.892e-16 | `physics.log`, `physics/physics_validation.json` |
| Historical corpus audit | PASS: duplicates/overlap absent, split coverage complete, nested subsets valid | `historical_data_audit.json` |
| Simple-baseline audit | 216 per-example rows, 24 summaries, 16 paired historical comparisons | `results/astra_baselines_20260927/` |
| Fourteen control paths, verification scale | 56/56 evaluation cells complete, 52 descriptive comparisons; repeated after source completion | `results/astra_controls_final_20260927/`, `controls_final.log` |
| Control corpus audit | PASS: 8 unlabeled, 16 labeled; disjoint splits | `control_data_audit.json` |
| Main runner, verification scale | 16/16 evaluation cells; resumed without repeating completed training/evaluation | `main_verification_resume.log`, run receipts |
| Incompatible resume probe | correctly rejected before training | `resume_rejection.json` |
| Original protocol/configuration preservation | byte-identical to starting HEAD | `frozen_protocol_check.json` |
| Manuscript metric linkage | 10 central metric values checked against saved summaries | `EVIDENCE_REGISTRY.json` |
| Technical manuscript | PDF compiled and all pages inspected | `paper/TECHNICAL_REPORT.pdf`, compilation receipt |
| CJSJ Word draft | official template, two manuscript pages plus figure page, rendered and inspected | `paper/submission/Krylov_paper_draft.docx` |
| Figure deck | editable native chart and caption, package/layout/data checks plus visual inspection | `paper/submission/Krylov_figures_checked.pptx` |
| Placeholder/secret-pattern/diff checks | no implementation placeholders or matching secret patterns; remaining TBDs are in superseded historical outline | scan receipts and `git diff --check` |

All receipt paths without a prefix are under `astra/2026-09-27/`. Result and paper
paths in this table are relative to `spectral-krylov-jepa/`. Small raw JSON evidence
is also copied under `run_receipts/`; large HDF5/checkpoints remain locally preserved
and ignored by Git. Raw control runs live under `experiments/raw/astra_controls_*`.
The bounded main run is `astra_main_verification_20260927T045810Z`.

## Results and their limits

| Method on historical test-ID | Labels | Fidelity | Residual at true energy |
|---|---:|---:|---:|
| Krylov-JEPA | 20 | 0.991650 | 40.190890 |
| Scratch | 20 | 0.993945 | 29.043635 |
| Fixed free-box shape | 0 | 0.994460 | 1.483805 |
| Nine-mode Ritz | 0 | 0.999898 | 0.761666 |
| Linear ridge | 20 | 0.999904 | 0.516766 |

These are actual retained computations on twelve ID test potentials. The paired
Krylov-minus-free-box fidelity difference is -0.002810, with conditional 95% CI
[-0.006733, 0.000182]; the interval crosses zero. Against Ritz, it is -0.008249
[-0.012187, -0.005594]. The inference is exploratory and conditional on one fitted
neural model, without seed uncertainty or multiplicity adjustment. Ritz has a
different inference cost and accesses the Hamiltonian. No universal method ranking
or equal-compute speedup follows.

Strong-amplitude OOD also remains unfavorable: retained Krylov relative energy
error 1.151349, versus Ritz 0.032930. Near-zero energies can amplify relative error.
Full residual and per-example evidence is retained rather than selecting one metric.

The control runs use grid eight, four pretrain steps and two fine-tune epochs.
They verify software paths, not learned performance, rank sensitivity or convergence.
Removing V leaves the transferred potential encoder untrained. Equal step counts
across the main neural methods do not match total parameters/FLOPs. The original
main runner's three fine-tune seeds share one pretrained encoder. These limitations
are explicit in both manuscripts and the claim ledger.

## Failures preserved and repairs

The first main-run and standalone physics plotting attempts aborted under the
macOS graphical Matplotlib backend. Failure logs remain as
`main_verification_backend_abort.log` and `physics_backend_abort.log`. The main
training/evaluation artifacts were preserved and resumed successfully using
`MPLBACKEND=Agg`; the numerical protocol was unchanged. Reproduction instructions
now set that backend. The first figure-deck validation rejected excessive literal
floating-point precision, then a missing runtime environment variable. Rounding
displayed figure data to ten decimals and supplying the documented runtime path
resolved those packaging failures. A reused receipt path was also rejected; the
builder now creates a fresh timestamped working directory for each invocation. Full-precision scientific JSON remains intact.
No scientific outcome was changed to make a check pass.

## Compute and external blockers

The historical “140 MB free” note was stale: the starting check found about 9.2 GiB
on the internal volume and 607 GiB on PRO-BLADE. Several other heavy portfolio jobs
were active, exceeding the attachment's recommended concurrency ceiling; this
sandbox reported MPS unavailable. `compute_preflight.json` records the observed
snapshot. The multi-hour main campaign and substantially larger supplement were
not launched; bounded checks ran sequentially with one CPU thread where configured.

Still required:

1. Execute and analyze the frozen main matrix (192 evaluations) and full supplement
   (672 evaluations) with retained failures and independent SSL-seed variation.
2. Complete representation-collapse/probe diagnostics, full-scale stability and
   sensitivity, and any strictly equal-total-compute efficacy comparison.
3. Reproduce on a clean environment/independent machine; hosted CI was defined but
   not run. No remote publication or external replication is claimed.
4. Complete human scientific review, real authorship/affiliations, CJSJ eligibility,
   consent/signatures and final AI-use disclosure. The live venue guidelines were
   verified; the source template's obsolete 2025 deadline text was not carried over.
5. Decide whether the bounded negative report warrants a submission independently
   of the unconfirmed positive hypothesis. No arXiv-ready label or upload is made.

## Reproduction and cleanup

Exact commands, environment choices, run/resume behavior and manuscript generation
are in `spectral-krylov-jepa/REPRODUCE.md`. The top-level
`COMPLETION_CHECKLIST_2026-09-27.md` maps every prompt-21 requirement to completed or
explicitly blocked work. `CLEANUP_MANIFEST.md` records the limited cleanup.

No user work, historical experiment, unique data or frozen protocol was removed.
No LMOP study was run or rewritten. The task produces a reviewable engineering and
negative-evidence checkpoint, not a finished positive research claim.
