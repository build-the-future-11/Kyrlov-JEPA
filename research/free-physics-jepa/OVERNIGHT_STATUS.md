# Free-Physics JEPA / LMOP-JEPA — Overnight Status

**Claim gate:** ENGINEERING ONLY (init)

| Phase | Status |
|-------|--------|
| 0 Repository/protocol audit | IN PROGRESS |
| 1 Darcy physics | NOT STARTED |
| 2 Data contract/manifests | NOT STARTED |
| 3 Genuine data | NOT STARTED |
| 4 Manufactured data | NOT STARTED |
| 5 Synthetic→real gap | NOT STARTED |
| 6 Shared backbone | NOT STARTED |
| 7 MML-direct | NOT STARTED |
| 8 LMOP-JEPA | NOT STARTED |
| 9 Shuffled control | NOT STARTED |
| 20 Smoke gate | NOT STARTED |
| 21 Confirmatory matrix | NOT STARTED |

Amendment: `amendments/PROTOCOL_AMENDMENT_0001_initial_freeze.md` (protocol files were absent; frozen from session brief).

Krylov-JEPA is unrelated and not used.

## Run smoke_20260926T150135Z
- mode: smoke
- device: mps
- grid: 32

- physics: PASSED cond≈2.27e+03
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/smoke_splits.json sha=907a3be89dc2
- pretrain MML-direct...
  MML done loss=0.4900 params=66417
- pretrain LMOP-JEPA...
  LMOP done loss=0.0116 var=1.538e-04
- shuffled-physics LMOP control...
- finetune scratch_n16_s11
- finetune mml_direct_n16_s11
- finetune lmop_jepa_n16_s11
- finetune scratch_n32_s11
- finetune mml_direct_n32_s11
- finetune lmop_jepa_n32_s11
- table: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/smoke_metrics.csv
- DONE smoke

## Run confirmatory_20260926T150251Z
- mode: confirmatory
- device: mps
- grid: 64

- physics: PASSED cond≈2.2e+04
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/confirmatory_splits.json sha=0610c3d24eab
- pretrain MML-direct...
  MML done loss=0.1288 params=1185057
- pretrain LMOP-JEPA...

## Run confirmatory_20260926T150845Z
- mode: confirmatory
- device: mps
- grid: 64

- physics: PASSED cond≈2.2e+04
- data: RESUME existing HDF5
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/confirmatory_splits.json sha=0610c3d24eab
- pretrain MML-direct...
  MML reused from prior confirmatory attempt
  MML done loss=0.12881408631801605 params=1185057
- pretrain LMOP-JEPA...
  LMOP done loss=0.8241 std=0.2936771755358204
- finetune scratch_n25_s11
- finetune mml_direct_n25_s11
- finetune lmop_jepa_n25_s11
- finetune scratch_n100_s11
- finetune mml_direct_n100_s11
- finetune lmop_jepa_n100_s11
- finetune scratch_n25_s23
- finetune mml_direct_n25_s23
- finetune lmop_jepa_n25_s23
- finetune scratch_n100_s23
- finetune mml_direct_n100_s23
