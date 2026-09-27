> Krylov status update (ASTRA, 2026-09-27): see `ASTRA_FINAL_REPORT.md` and `STATUS.md`. The historical audit below is preserved; LMOP status was not changed by this execution.

# Master execution queue

**Updated:** 2026-09-27. Status: DONE · OPEN · BLOCKED (reason) · NOT RUN

## P0 — critical

| Task | Status |
|---|---|
| Degenerate across-seed CI counted as LMOP zero-shot "win": fix gate, re-gate, withdraw claim | DONE (`NOT_TESTABLE_NO_SEED_VARIANCE`) |
| Study-3 runs overwrote canonical Study-2 claim gate / `RESULTS_AUTO.md` (wrong source CSV) | DONE (restored; per-study output stems) |
| "Breakthrough" publish stance / commit message overclaim | DONE (checklist + draft corrected; commit history left intact) |
| Trivial-baseline audit: all trained models worse than train-mean field | DONE (documented; root cause identified) |
| Decide whether to run Study-4 (amendment 0005, coordinate channels) | BLOCKED (author approval; forking-paths decision) |
| Internal disk full (140 MB free) | BLOCKED (user action; nothing deleted automatically) |

## P1 — must finish

| Task | Status |
|---|---|
| Finish Study-3 `low_lr_ft` matrix (13/18 → 18/18) | DONE (`FALSIFIES_HYPOTHESIS`, uninformative) |
| Make `eval_study3.py` resumable | DONE (`--resume-run`) |
| Study-3 summary + figures from all CSVs | DONE (`scripts/summarize_study3.py`) |
| Paper draft consistent with artifacts | DONE (`paper/DRAFT.md`; references section still empty, no fabricated citations) |
| Krylov decisive script aligned with frozen protocol (λ_E, OOD splits, paths, resume) | DONE (sanity run passed; artifacts removed) |
| Krylov decisive experiment | NOT RUN (≈3–5 h MPS; waiting on disk space + go-ahead) |
| Krylov declared ablations runner (K1–K3, shuffled physics, remove-V) | OPEN |
| Put `spectral-krylov-jepa/` under version control (exclude `.venv`, raw data) | DONE (tracked in workspace repo) |

## P2 — high value

| Task | Status |
|---|---|
| Commit LMOP run/data provenance (4.3 MB JSON metrics + meta; checkpoints/HDF5 already ignored) | DONE |
| Real bibliography for LMOP draft (primary sources only) | OPEN |
| Pin Study-2 λ values separately from `protocol.yaml` so Study-2 reproduces from HEAD | OPEN (documented: use `git checkout 1f0430d`) |
| Claim-gate unit tests (`tests/test_claim_gate.py`, 4 tests) | DONE |
| CI workflow (no remote configured, so nothing to run it) | OPEN |

## P3 — optional

| Task | Status |
|---|---|
| Separate E_u tower (amendment 0004 §E) | Deferred; moot until the architecture is repaired |
