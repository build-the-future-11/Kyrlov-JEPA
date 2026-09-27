# Kyrlov workspace status — 27 September 2026

The ASTRA prompt-21 execution applies to `spectral-krylov-jepa/`. The neighboring
LMOP-JEPA project was preserved; its Study-4 decision remains outside this task.

| Area | Current state |
|---|---|
| Engineering | 39 tests pass; critical Python lint and compilation pass; 14 controls execute at verification scale |
| Science | Negative baseline audit of the historical smoke corpus; no demonstrated Krylov transfer gain |
| Main frozen experiment | NOT RUN: shared local compute contention; 192 evaluations configured |
| Full supplemental controls | NOT RUN: shared local compute contention; 672 evaluations configured |
| Paper | Technical report and CJSJ template draft compiled/rendered; human scientific review required |
| Release/submission | BLOCKED: full experiment evidence, independent reproduction, author eligibility/consent and final review |

See `ASTRA_FINAL_REPORT.md`, `COMPLETION_CHECKLIST_2026-09-27.md`, and
`spectral-krylov-jepa/REPRODUCE.md`. High neural fidelity on the retained development
set is insufficient: a fixed free-box shape scores higher in mean, and linear and
Ritz controls score better still. This is a bounded historical result, not proof
that all Krylov representations fail. Existing frozen results were not retuned.
