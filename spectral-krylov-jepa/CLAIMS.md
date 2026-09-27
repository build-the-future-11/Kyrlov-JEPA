# Claims and evidence — 27 September 2026

| Claim | Evidence | Classification |
|---|---|---|
| Physics and core execution paths work locally | `../astra/2026-09-27/tests.log`, `physics.log`, critical lint receipt | VERIFIED engineering only |
| Historical data splits are disjoint, subsets nested, unlabeled corpus separate | `../astra/2026-09-27/historical_data_audit.json` | VERIFIED content audit |
| Fixed sine shape has greater mean ID fidelity than retained Krylov N20 | `results/astra_baselines_20260927/baseline_audit.json` paired rows | OBSERVED exploratory; conditional CI crosses zero |
| Nine-mode Ritz outperforms retained Krylov on ID fidelity and physical residual | same JSON plus `experiments/summaries/smoke_latest.json` | OBSERVED exploratory historical comparison |
| Linear ridge is a strong baseline on the retained family | `baseline_summary.csv`, all rows retained | OBSERVED exploratory |
| Krylov pretraining improves few-shot transfer | original full matrix missing | UNESTABLISHED |
| Orthogonalization, coefficient loss or rank helps learning | full supplemental matrix missing | NOT TESTED scientifically |
| Fourteen variants execute | `results/astra_controls_20260927/analysis.json` and raw metrics | VERIFIED pipeline, not efficacy |
| Equal training steps establish compute equality | parameter/token counts differ | FALSE as a general inference |
| Normalized powers provide a matched neural architecture control | `12_run_controls.py`, K2/powers parameter counts | IMPLEMENTED; total data-generation cost not matched |
| Remove-V isolates only potential information | transferred potential encoder is untrained | NOT A CLEAN MECHANISM ESTIMATE |
| Krylov basis spans powers in exact arithmetic without breakdown | `paper/TECHNICAL_REPORT.md`, Proposition 1 proof | CLASSICAL FACT; not neural novelty |
| Small infidelity implies small physical residual without a spectral bound | Proposition 2 counterexample | FALSE |
| Release, independent reproduction or journal acceptance | no such evidence | BLOCKED / NOT RUN |

The main design has one SSL seed and three downstream seeds; it cannot quantify
independent pretraining-seed uncertainty. The full supplementary runner uses three
independent SSL seeds but is not executed. Finite-test bootstrap intervals are
conditional, exploratory and unadjusted. No SOTA, quantum advantage, foundation
model or universal transfer claim is supported.
