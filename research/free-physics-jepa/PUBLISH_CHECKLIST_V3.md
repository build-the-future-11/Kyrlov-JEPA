# Study-3 publish checklist (amendment 0004)

**Status (2026-09-27 audit): NOT PUBLISHABLE AS A TRANSFER RESULT.** See `../../RESEARCH_AUDIT.md`.

The previous "breakthrough / zero-shot LMOP win" stance is withdrawn:

1. **Zero-shot gate was degenerate.** No fine-tuning occurs, so MML and LMOP predictions are identical across seeds and N; the across-seed CI has zero width and any gap counts as "non-overlapping". Re-gated with the fixed logic: `NOT_TESTABLE_NO_SEED_VARIANCE`.
2. **Zero-shot was post-hoc.** Amendment 0004 (written 18:50) quotes the exact zero-shot numbers it then treated as confirmatory; the protocol choice followed the observation.
3. **Zero-shot is practically meaningless here.** LMOP zero-shot (0.686 ID) is worse than Scratch fine-tuned on 25 labels under every fine-tune protocol (0.515–0.538 ID).
4. **All Study 1–3 models fail a trivial baseline.** Every trained model (0.50–0.54) is worse than the input-blind train-mean field (0.33 ID / 0.24 OOD). Root cause: no coordinate channels in `FNO2d` (see amendment 0005, PROPOSED).

## Done

- [x] Amendment 0004 frozen (zero_shot + probe primary; λ_var=1, λ_u=1)
- [x] Eval harness `scripts/eval_study3.py` (now resumable; per-adaptation gates no longer overwrite Study-2 files)
- [x] Zero-shot matrix 36/36 — `NOT_TESTABLE_NO_SEED_VARIANCE` (exploratory/post-hoc)
- [x] LMOP retrained with λ_var=1, λ_u=1; shuffle control supported (2.23 > 1.40)
- [x] Probe matrix 36/36 — `FALSIFIES_HYPOTHESIS` (uninformative, see item 4)
- [x] Low-LR fine-tune matrix 36/36 — `FALSIFIES_HYPOTHESIS` (uninformative, see item 4)
- [x] Summary + figures: `STUDY3_RESULTS.md`, `figures/confirmatory_study3_adaptations.png`, `figures/audit_trivial_baselines.png`
- [x] Paper draft updated with the audited interpretation

## Open (author decision)

- [ ] Approve or reject amendment 0005 (coordinate channels, Study-4). Without it, the research question remains untested.
- [ ] If rejected: publish only as a methods/negative note on evaluation pitfalls (trivial baselines, degenerate seed CIs), not as evidence about LMOP vs MML.

## Do not

- Claim any LMOP advantage from zero-shot numbers.
- Describe Studies 1–3 as falsifying the hypothesis in a scientific sense; say the comparison was uninformative.
- Pool results across studies.
