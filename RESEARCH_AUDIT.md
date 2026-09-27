> Krylov status update (ASTRA, 2026-09-27): see `ASTRA_FINAL_REPORT.md` and `STATUS.md`. The historical audit below is preserved; LMOP status was not changed by this execution.

# Research audit — Kyrlov-JEPA workspace

**Date:** 2026-09-27 · **Scope:** `research/free-physics-jepa` (LMOP-JEPA), `spectral-krylov-jepa` (Krylov-JEPA)  
Supersedes `AUDIT_LMOP_JEPA.md` (2026-09-26), which was written before any LMOP code existed.

Statuses: VERIFIED · PARTIALLY VERIFIED · UNSUPPORTED · FAILED · NOT RUN · BLOCKED

## 1. LMOP-JEPA (Darcy, manufactured-solution transfer)

### Result generations

| Generation | Amendments | Label | Artifacts |
|---|---|---|---|
| Study-1 | 0001–0002 | historical / known-broken pipeline (MMS scale mismatch, broken shuffle) | `results/tables/confirmatory.csv` |
| Study-2 | 0003 | preregistered; **uninformative** (trivial-baseline failure) | `confirmatory_v2.csv`, `claim_gate_v2.md` |
| Study-3 zero_shot | 0004 | **post-hoc / exploratory**; degenerate gate | `study3_confirmatory_zero_shot.csv`, `claim_gate_s3_zero_shot.md` |
| Study-3 probe, low_lr_ft | 0004 | preregistered; **uninformative** (trivial-baseline failure) | `study3_confirmatory_{probe,low_lr_ft}.csv`, `claim_gate_s3_*.md` |
| Study-4 | 0005 | PROPOSED, **NOT RUN** | `amendments/PROTOCOL_AMENDMENT_0005_coordinate_channels.md` |

### Claims

| Claim | Evidence exists? | Artifact | Reproducible? | Status |
|---|---|---|---|---|
| Discrete Darcy operator is identical for manufacture and genuine solves; physics check passes | Yes | `darcy_reference.py`, `runs/*/physics_validation.json` | Yes | VERIFIED |
| No split leakage (train/val/test/OOD/manufactured disjoint; N25 ⊂ N100) | Yes (SHA-1 of every `a` field, 2026-09-27) | `manifests/confirmatory_v2_splits.json`, `data/confirmatory_v2/` | Yes | VERIFIED |
| Model selection uses validation only | Yes (code) | `lmop_jepa/training/__init__.py::finetune_genuine` | Yes | VERIFIED |
| Manufactured/genuine norm matched (ratio 0.978) | Yes | `confirmatory_v2_synthetic_real_mismatch.json` | Yes | VERIFIED |
| Shuffled-physics control worse than correct (S2: 15.83>14.58; S3: 2.23>1.40) | Yes | `runs/*/shuffled_control.json` | Yes | VERIFIED |
| "LMOP-JEPA beats MML zero-shot with non-overlapping CIs" (commit `22a0696`) | Numbers exist; inference invalid | `claim_gate_s3_zero_shot.md` (re-gated) | Yes | **UNSUPPORTED**: zero-width seed CI, post-hoc protocol, worse than Scratch N=25 |
| "MML ≥ LMOP" (Study-2, Study-3 probe/low_lr frozen gates) | Yes, as executed | `claim_gate_v2.md`, `claim_gate_s3_probe.md`, `claim_gate_s3_low_lr_ft.md` | Yes | PARTIALLY VERIFIED: gate executed correctly, but **not scientifically informative** (below) |
| Downstream models learn genuine Darcy solutions better than a trivial baseline | Checked | `results/tables/audit_positional_floor.json` | Yes | **FAILED**: all trained models 0.50–0.54 vs train-mean field 0.33 ID / 0.24 OOD |
| Root cause: no coordinate channels in `FNO2d` (translation-equivariant, cannot locate Dirichlet boundary) | Yes | `audit_positional_floor.json` (boundary/center 0.96–1.00 vs truth 0.06); `audit_coord_sanity.json` (val 0.509 → 0.084 with coords) | Yes (single seed, validation only) | PARTIALLY VERIFIED (engineering evidence, not claim evidence) |
| Input normalization makes target scale unidentifiable | Checked | `audit_scale_identifiability.json` | Yes | Ruled out as main cause (floor ≈0.03) |
| LMOP transfers better/worse than MML (research question) | No valid test | — | — | **NOT RUN** (requires amendment 0005 / Study-4) |
| Study-2 reproducible from HEAD | No | `protocol.yaml` now holds Study-3 λ values | Only via `git checkout 1f0430d` | PARTIALLY VERIFIED |

### Integrity defects found and fixed (2026-09-27)

1. **Degenerate CI gate.** `aggregate_claim_gate.py` treated zero-width across-seed intervals as non-overlapping. Added `seed_variance_degenerate` and the verdict `NOT_TESTABLE_NO_SEED_VARIANCE`. Re-gating Study-2, probe, and low_lr_ft gives unchanged verdicts. Zero-shot changes from SUPPORTS to NOT_TESTABLE. The original verdict is preserved in git history (`22a0696`).
2. **Artifact clobbering.** Study-3 runs overwrote the canonical `claim_gate.{json,md}` and `paper/RESULTS_AUTO.md` with probe results and cited the wrong source CSV and amendment list. I restored the Study-2 versions, and the gate now writes to per-study output names.
3. **Non-resumable eval.** `eval_study3.py` always created a new run directory, and the probe/low_lr run died after 13 of 18 low_lr fine-tunes. Added `--resume-run`; the matrix is now complete (36/36).
4. **Unseeded zero-shot Scratch init** (irreproducible Scratch zero-shot rows). Now seeded. The existing rows were not regenerated because Scratch is not part of the decisive comparison.
5. **Partial summaries.** `STUDY3_RESULTS.md` was rewritten by each invocation with only that invocation's adaptations. Replaced by `scripts/summarize_study3.py`, which reads every CSV on disk and labels each evidence class.
6. **Overclaiming docs.** The "breakthrough" stance in `PUBLISH_CHECKLIST_V3.md` is withdrawn, and `paper/DRAFT.md` is rewritten around the audited interpretation.

## 2. Spectral Krylov-JEPA (Schrödinger ground states)

| Claim | Evidence exists? | Artifact | Reproducible? | Status |
|---|---|---|---|---|
| FD Hamiltonian / Lanczos / eigensolver correct | Yes | `tests/` (23 passed, 2026-09-27), `scripts/00_validate_physics.py` | Yes | VERIFIED |
| Smoke pipeline runs end-to-end (4 methods) | Yes | `results/tables/smoke_metrics.csv`, `experiments/raw/smoke_suite_20260926T074651Z` | Yes | VERIFIED (pipeline only; not evidence) |
| Krylov-JEPA improves few-shot ground-state prediction | No | — | — | **NOT RUN** (decisive experiment never executed) |
| Decisive script matches frozen protocol | Was not | `scripts/11_run_main_experiment.py` | — | Fixed 2026-09-27: λ_E 0.2 → 1.0 (protocol value), all OOD splits evaluated, repo-anchored paths, resumable; pipeline sanity run passed and was deleted |
| Declared ablations (depth K1–K3, shuffled physics, remove-V, coeff loss) | No | — | — | NOT RUN (no runner script) |
| Version control | Yes (2026-09-27) | tracked in workspace repo; `.venv`, raw data, checkpoints excluded | — | VERIFIED |
