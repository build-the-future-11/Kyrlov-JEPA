> Updated evidence: see `CLAIMS.md`, `REPRODUCE.md` and `paper/TECHNICAL_REPORT.md` at this project root. The 2026-09-27 baseline audit found no established Krylov gain. Full experiments remain unrun. Historical content below is preserved.

# Results Status

**Last updated:** 2026-09-26 (after improved smoke suite)

## Completed (code + verification)

| Item | Status | Artifact |
|------|--------|----------|
| Physics / Lanczos | verified | `00_validate_physics.py` |
| `pytest` | **23 passed** | |
| Improved smoke (4 methods × N∈{10,20} + OOD) | **passed (~74s)** | `experiments/raw/smoke_suite_20260926T074651Z/` |
| Metrics table | real | `results/tables/smoke_metrics.csv` |
| Figures | real | `figures/label_efficiency_smoke.png`, `ood_comparison_smoke.png`, prediction panels |

## Improvements landed (this iteration)

- Energy/potential **subset-only** z-score standardization
- Stronger energy head + token-aware ψ decoder + sign-aligned MSE aux
- Early stop on \((1-F)+0.5\cdot\mathrm{relE}\)
- Dual residuals: at \(\hat E\), at true \(E_0\), at Rayleigh \(E_R\)
- Smoke now runs Scratch / Field / Operator / Krylov + Krylov OOD

## Real improved-smoke test-ID numbers (seed=11, 16×16)

| Method | N | Fidelity | Rel. E (head) | Rel. E (Rayleigh) | Residual @ true E |
|--------|---|----------|---------------|-------------------|-------------------|
| Scratch | 10 | 0.983 | **0.033** | 0.460 | 45.0 |
| Scratch | 20 | 0.994 | **0.032** | 0.186 | 29.0 |
| Field | 10 | 0.986 | 0.038 | 0.508 | 50.4 |
| Field | 20 | 0.991 | 0.043 | 0.336 | 41.4 |
| Operator | 10 | 0.983 | 0.049 | 0.572 | 54.5 |
| Operator | 20 | 0.992 | 0.040 | 0.292 | 39.0 |
| Krylov | 10 | 0.980 | 0.036 | 0.715 | 61.3 |
| Krylov | 20 | 0.992 | 0.034 | 0.308 | 40.2 |

### vs previous smoke
Energy-head relative error dropped from ~**0.46** → ~**0.03–0.05** (standardization + stronger loss). Fidelity remains saturated on this tiny grid.

### Krylov OOD (N=20)
| Split | Fidelity | Rel. E head |
|-------|----------|-------------|
| test_ID | 0.992 | 0.034 |
| OOD_narrow | 0.988 | 0.034 |
| OOD_double | 0.993 | 0.055 |
| OOD_strong | **0.966** | **1.15** |

Strong-amplitude OOD clearly hurts energy prediction — useful stress signal.

## Scientific reading (smoke only — not a claim)

- **No evidence yet** that Krylov beats Scratch/Field/Operator on this smoke scale (fidelities within noise; Scratch often best).
- **Energy head fixed** enough to separate methods later; residual@true-E still large because \(F\approx0.98\) can leave high-energy admixtures that dominate \(\|H\psi\|\).
- Rayleigh energy error tracks residual better than the energy head — ψ quality remains the bottleneck.

## Not yet executed

- Decisive 32×32 / 1k unlabeled / N∈{10,25,50,100} × seeds {11,23,47}
- Full ablations (depth, shuffle, remove-V, coeff)
- `results/tables/label_efficiency.csv` for the decisive protocol
