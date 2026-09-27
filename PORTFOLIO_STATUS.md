# Portfolio status — /Volumes/PRO-BLADE/Kyrlov-JEPA

**Updated:** 2026-09-27 · Git: single repo at workspace root, branch `master`, no remote, no CI (`.github/` absent). No secrets found (pattern scan of source/docs).

| Project | Path | Type | Current State | Critical Blocker | Evidence | Next Action | Priority |
|---|---|---|---|---|---|---|---|
| LMOP-JEPA (Free-Physics JEPA) | `research/free-physics-jepa/` | ML research / scientific computing | Experimental → negative-methods manuscript draft. Studies 1–3 complete (Study-3: zero_shot, probe, low_lr_ft all 36/36) | Downstream FNO lacks coordinate channels, so every trained model is worse than the train-mean-field baseline and the research question is untested | `RESEARCH_AUDIT.md`, `results/tables/audit_*.json`, `STUDY3_RESULTS.md` | Author decision on amendment 0005 (Study-4, ≈4–5 h MPS) | P0 decision |
| Spectral Krylov-JEPA | `spectral-krylov-jepa/` | ML research (Schrödinger eigenstates) | Implementation, smoke-verified; 23/23 tests pass | Decisive experiment NOT RUN (≈3–5 h MPS) | `paper/results_status.md`, `results/tables/smoke_metrics.csv` | Run `scripts/11_run_main_experiment.py` (now protocol-aligned, resumable); then add ablation runner | P1 |

## Environment notes

- Internal disk has ≈140 MB free (`/System/Volumes/Data` 100% full). All project data/runs live on PRO-BLADE (609 GB free). Temp files for agent work were redirected to `/Volumes/PRO-BLADE/.agent-tmp`. **Free internal disk space before long runs**; macOS may fail to swap or write logs.
- Python env for both projects: `spectral-krylov-jepa/.venv` (Python 3.14.7, torch 2.14.0, MPS).
