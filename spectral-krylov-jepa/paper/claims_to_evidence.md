> Updated evidence: see `CLAIMS.md`, `REPRODUCE.md` and `paper/TECHNICAL_REPORT.md` at this project root. The 2026-09-27 baseline audit found no established Krylov gain. Full experiments remain unrun. Historical content below is preserved.

# Claims → Evidence

| Claim | Required evidence | Status |
|-------|-------------------|--------|
| Krylov pretraining improves label efficiency vs scratch | Matched downstream model; frozen nested subsets N=10/25/50/100; ≥3 seeds; held-out ID test; paired bootstrap | **unsupported at smoke scale** (Scratch ≥ Krylov); decisive run not executed |
| Krylov beats Field-JEPA at low N | Same labeled data/splits; matched steps/batch/arch budget | **unsupported at smoke scale** (Field ≈ Krylov); decisive run not executed |
| Krylov beats Operator-JEPA at low N | Same as above | **unsupported at smoke scale**; decisive run not executed |
| Operator-aware pretraining degrades less under OOD structural shift | OOD_narrow / OOD_strong / OOD_double; no OOD retuning | **mixed / smoke only** — OOD_strong hurts Krylov energy badly (relE≈1.15); ID/narrow/double OK |
| Model uses Hamiltonian info (needs V) | `remove_v` ablation underperforms matched Krylov | **not tested** |
| Model uses physics-consistent dynamics | Shuffled-(V,trajectory) control underperforms | **not tested** |
| Deeper Krylov helps until saturation | K1/K2/K3 depth ablation | **not tested** |
| α/β auxiliary loss helps | JEPA-only vs JEPA+coeff | **not tested** |
| Krylov-JEPA is a foundation model / SOTA / always reduces sample complexity | — | **out of scope / disallowed** |

Update status only when corresponding runs and artifacts exist under `experiments/raw/` and `results/`.
