# Integrity finding — Study-3 "zero-shot LMOP win" (2026-09-27)

Read-only audit. No results, protocols or scripts in this project were changed.

## Finding

The claim in commit `22a0696` ("Land Study-3 amendment 0004 with zero-shot LMOP win")
is **UNSUPPORTED** as confirmatory evidence.

1. **Amendment 0004 is post hoc.** It calls itself "frozen a priori", but its Problem
   section cites Study-2 genuine `test_id` outcomes (zero-shot LMOP 0.69 < MML 0.78, and
   MML winning every fine-tuned cell). It then designates `zero_shot` a primary claim
   protocol and reuses the same `data/confirmatory_v2/` test data (section D).
2. **The Study-3 zero-shot numbers are the pre-amendment observation.**
   `results/tables/claim_gate_s3_zero_shot.md` reports LMOP 0.6857 vs MML 0.7778 (ID) —
   the values already quoted from Study-2 when the amendment was written. Re-evaluating
   the same checkpoints on the same test set is not an independent test.
3. **No seed variance.** MML and LMOP each use a single pretrained checkpoint, so the
   "non-overlapping seed CI" rule is degenerate. The working-tree gate now reports
   `NOT_TESTABLE_NO_SEED_VARIANCE`, which is correct but understates item 1.
4. **Flexible success rule.** Support requires a win on any one of roughly eight cells
   ({zero_shot, probe} × N∈{25,100} × {ID, OOD}), without multiplicity control.
5. `AUDIT_LMOP_JEPA.md` at the repository root predates this program's creation and
   says it is absent; it is stale.

## Legitimate status

- Study-1 and Study-2: archived negative results under their protocols (retain).
- Zero-shot LMOP < MML relative L2 on `test_id`: an **exploratory observation** from
  Study-2, one checkpoint per method.
- Confirmation would need a protocol frozen before new outcomes, fresh held-out
  genuine solves never used for amendment decisions, multiple independently pretrained
  checkpoints per method, and a single prespecified primary cell or multiplicity control.

## Current working tree (checked 2026-09-27)

The uncommitted working tree already withdraws the win (`PUBLISH_CHECKLIST_V3.md`) and
labels the zero-shot table EXPLORATORY / POST-HOC (`paper/RESULTS_STUDY3.md`), consistent
with this finding. Those corrections are **not committed**; `HEAD` still carries the
claim in its commit message and committed files.

## Recommended owner actions

- Commit the working-tree withdrawal so the history no longer ends on the unsupported claim.
- Apply the same pre-registration scrutiny to untracked amendment 0005 before it runs.
