# Study-2 publish checklist (amendment 0003)

Study-1 is archived (negative/null). This checklist is for **winning or honestly reporting Study-2**.

## Already done

- [x] Autopsy of Study-1 failure modes
- [x] `PROTOCOL_AMENDMENT_0003_matched_block_jepa.md` frozen
- [x] Protocol.yaml Study-2 knobs (matched MMS, VICReg, freeze schedule, `u_vs_af`)
- [x] Same-encoder multi-block LMOP + full transfer
- [x] Input normalization + matched MMS generation (`data/*_v2/`)
- [x] Fixed shuffle control
- [x] Freeze-then-unfreeze fine-tune
- [x] Smoke v2 gate passed (`smoke_s2_v2_*`, shuffle supported=True)
- [x] Overnight entrypoint `scripts/overnight_confirmatory_v2.sh`
- [x] Git SHA baseline (`0256cb6`)

## Do now

- [ ] Run confirmatory Study-2 matrix (fresh `data/confirmatory_v2/`)
- [ ] Confirm matrix complete: **36/36** eval rows in `results/tables/confirmatory_v2.csv`
- [ ] Confirm shuffle gate written (`runs/.../shuffled_control.json`)
- [ ] Read `results/tables/claim_gate_v2.md` verdict
- [ ] Confirm figures: `figures/confirmatory_v2_label_efficiency.png`
- [ ] Confirm `paper/RESULTS_AUTO.md` regenerated
- [ ] Record git SHA + run id in overnight status
- [ ] Commit confirmatory tables / claim gate (not huge `.pt` / `.h5`)

## Morning / publish writeup

- [ ] Paste `RESULTS_AUTO.md` into `paper/DRAFT.md` §Results
- [ ] State Study-1 as negative archive; Study-2 as amended protocol
- [ ] Use claim-gate language only (support / falsify / inconclusive)
- [ ] Do **not** cite smoke or Krylov-JEPA as LMOP evidence

## Success criteria (unchanged)

LMOP beats MML on relative \(L_2\) with non-overlapping seed CIs at \(N=25\) and/or \(N=100\`, **and** shuffled loss > correct loss.
