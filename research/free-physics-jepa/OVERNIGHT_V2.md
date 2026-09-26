# Study-2 overnight (amendment 0003)

Study-1 (`confirmatory_20260926T150845Z`) is archived as a **negative/null** result.

Study-2 fixes: matched MMS norms, input normalization, same-encoder multi-block JEPA + VICReg, fixed `u_vs_af` shuffle, freeze-then-unfreeze fine-tune.

## Smoke first (recommended, ~minutes)

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=research/free-physics-jepa:research/free-physics-jepa/lmop_jepa:research/free-physics-jepa/scripts \
./spectral-krylov-jepa/.venv/bin/python \
  research/free-physics-jepa/scripts/run_experiment.py --mode smoke
```

Check `results/tables/smoke_v2_synthetic_real_mismatch.json` — `norm_ratio_mfg_over_genuine` should be ~1.

## Confirmatory overnight

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
./research/free-physics-jepa/scripts/overnight_confirmatory_v2.sh
```

Resume a partial Study-2 run:

```bash
./research/free-physics-jepa/scripts/overnight_confirmatory_v2.sh confirmatory_s2vv2_<timestamp>
```

## Morning artifacts

- `results/tables/confirmatory_v2.csv`
- `results/tables/claim_gate_v2.md`
- `paper/RESULTS_AUTO.md`
- `figures/confirmatory_v2_label_efficiency.png`
