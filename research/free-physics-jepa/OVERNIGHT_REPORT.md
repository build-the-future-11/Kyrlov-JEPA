# Free-Physics JEPA / LMOP-JEPA — Overnight Report

**Generated:** 2026-09-26  
**Workspace:** `/Volumes/PRO-BLADE/Kyrlov-JEPA`  
**Claim gate at time of writing:** see bottom (updated as runs finish)

## Repository state

- Created `research/free-physics-jepa/` as the LMOP program (distinct from `spectral-krylov-jepa/`).
- Protocol files were **absent**; authored and frozen via `amendments/PROTOCOL_AMENDMENT_0001_initial_freeze.md`.
- No git repository at workspace root (cannot record SHA yet).

## Commits

None (workspace not a git repo).

## Tests executed

```bash
python3 -m pytest tests/test_free_physics_jepa_darcy.py -v
```

**Result: 9 passed** (symmetry, SPD, round-trip modes, rejection, conditioning, etc.)

## Data generated (smoke)

Under `research/free-physics-jepa/data/smoke/`:

- genuine train/val/test_id/ood HDF5
- manufactured mixed + low HDF5
- manifests: `manifests/smoke_splits.json`

## Methods implemented

- Scratch (shared FNO2d downstream)
- MML-direct (manufactured regression → fine-tune)
- LMOP-JEPA (EMA target JEPA on manufactured → transfer trunk → fine-tune)
- Shuffled-physics LMOP control (smoke)

## Runs completed

### Smoke (`runs/smoke_20260926T150135Z/`) — **NOT SCIENTIFIC EVIDENCE**

Matched params ≈66k. Pretrain 100 steps.

| Method | N | Dist | relative_L2 |
|--------|---|------|-------------|
| Scratch | 16 | id | 0.492 |
| MML | 16 | id | **0.480** |
| LMOP | 16 | id | 0.529 |
| Scratch | 32 | id | 0.476 |
| MML | 32 | id | **0.476** |
| LMOP | 32 | id | 0.490 |

Artifacts: `results/tables/smoke_metrics.csv`, `figures/smoke_label_efficiency.png`, `figures/smoke_synthetic_real_norms.png`.

### Shuffled control (smoke)

Pretrain loss correct ≈0.012 vs shuffled ≈0.012 (nearly identical at 100 steps). **Mechanism not supported at smoke scale** — must re-check at confirmatory compute.

### Synthetic→real gap (diagnostic)

Genuine mean ‖u‖₂ ≈ 1.29; manufactured mixed ≈ 9.88 (large scale mismatch). Documented; **not** used to redesign corpus mid-study.

## Runs incomplete / in flight

- Confirmatory matrix `{Scratch,MML,LMOP} × {25,100} × {11,23,47}` on 64×64 — launched after smoke.

## Scientific interpretation so far

Smoke only: **MML-direct ≤ LMOP on ID relative L2** (LMOP does not win). This does **not** close the confirmatory question.

## Protocol amendments

1. `PROTOCOL_AMENDMENT_0001_initial_freeze.md` — create freeze because referenced files were missing.

## Memory/compute

- `num_workers=0`, OMP/MKL threads=4
- Smoke peak comfortable on laptop; confirmatory uses batch_size=4, float32 train

## Remaining blockers

- Overnight confirmatory resume must finish remaining 8 finetunes + shuffled control

## Exact next command

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
./research/free-physics-jepa/scripts/overnight_confirmatory.sh
```

See `OVERNIGHT.md` / `PUBLISH_CHECKLIST.md`.

## Claim gate

**SMOKE VERIFIED** (pipeline integrity only).

Not yet: CONFIRMATORY RESULTS AVAILABLE / CJSJ EVIDENCE READY — produced automatically at end of overnight run into `results/tables/claim_gate.json`.
