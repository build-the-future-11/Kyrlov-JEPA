# Spectral Krylov-JEPA

**Krylov-Subspace Joint-Embedding Pretraining for Few-Shot Quantum Eigenstate Prediction**

This repository tests a single, falsifiable question:

> Do short, inexpensive Lanczos trajectories provide a more transferable self-supervised representation for few-shot ground-state prediction than potential-field masking or a single Hamiltonian action?

We study the 2D stationary Schrödinger equation

\[
H_V \psi = E\psi, \qquad H_V = -\tfrac12\nabla^2 + V(x,y)
\]

on the unit square with Dirichlet boundaries. Expensive labels are ground-state eigenpairs \((E_0,\psi_0)\). Cheap pretraining signals are Hamiltonian–vector products and short Lanczos trajectories.

**This is not a foundation-model claim.** Negative or null results are first-class outcomes.

---

## Comparison hierarchy

| Method | Pretraining signal |
|--------|--------------------|
| Scratch | None |
| Field-JEPA | Masked potential patches |
| Operator-JEPA | One action: \((V,q)\to z(\mathrm{normalize}(Hq))\) |
| **Spectral Krylov-JEPA** | Short Lanczos: \((V,q_0,q_1)\to z(q_2)\) |

All methods share the same labeled splits, downstream architecture budget, optimizer family, and evaluation code.

---

## Installation

```bash
cd spectral-krylov-jepa
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Requires Python ≥ 3.10. Runs on CPU by default; CUDA/MPS used automatically when available.

---

## Quickstart (smoke suite)

One command runs physics validation → data generation → Krylov pretraining → scratch + Krylov fine-tuning → evaluation → real metrics/figures:

```bash
python scripts/10_run_smoke_suite.py
```

Expected artifacts:

- `figures/figure_A_physics_panel.png`
- `results/tables/smoke_metrics.csv`
- `experiments/summaries/smoke_latest.json`
- run directory under `experiments/raw/smoke_suite_*`

---

## Physics validation

```bash
python scripts/00_validate_physics.py
```

Fails loudly if symmetry, residual, or Lanczos tolerances are violated. **No ML should run before this passes.**

---

## Generate data

```bash
# Unlabeled Krylov corpus (prototype: 1k; serious: 5k–10k)
python scripts/01_generate_unlabeled.py --n-examples 1000 --output experiments/raw/unlabeled.h5

# Labeled eigenstates + frozen splits
python scripts/02_generate_labeled.py --output experiments/raw/labeled.h5
```

---

## Pretrain

```bash
python scripts/03_pretrain_field_jepa.py --data experiments/raw/unlabeled.h5
python scripts/04_pretrain_operator_jepa.py --data experiments/raw/unlabeled.h5
python scripts/05_pretrain_krylov_jepa.py --data experiments/raw/unlabeled.h5
```

---

## Fine-tune & evaluate

```bash
python scripts/06_finetune.py --method scratch --subset n10 \
  --data experiments/raw/labeled.h5 --manifest experiments/manifests/labeled_splits.json

python scripts/06_finetune.py --method krylov --subset n10 \
  --encoder experiments/raw/<pretrain_run>/encoder.pt \
  --data experiments/raw/labeled.h5 --manifest experiments/manifests/labeled_splits.json

python scripts/07_evaluate.py --checkpoint experiments/raw/<finetune_run>/checkpoint_best.pt \
  --split test_ID --output results/metrics/<run>
```

---

## Reproduce main experiment

Frozen protocol: `paper/experiment_protocol.md`  
Config sketch: `configs/smoke/main_experiment.yaml`

```bash
# Full decisive experiment (NOT yet run): data → 3 pretrains → 4 methods × N∈{10,25,50,100} × seeds {11,23,47}
# → eval on test_ID + all OOD splits → results/tables/label_efficiency{,_per_seed}.csv + figures/label_efficiency*.png
python scripts/11_run_main_experiment.py --config configs/smoke/main_experiment.yaml
# Resume an interrupted run:
python scripts/11_run_main_experiment.py --config configs/smoke/main_experiment.yaml --resume-run main_label_efficiency_<timestamp>
```

---

## Results status

| Stage | Status |
|-------|--------|
| Physics engine | Implemented + validated |
| Lanczos generator | Implemented + validated |
| Dataset generation | Implemented |
| Models (Scratch / Field / Operator / Krylov) | Implemented |
| Unit/integration tests | **23 passed** |
| Smoke suite | **Executed** — see `paper/results_status.md` |
| Full label-efficiency experiment | **Not yet executed** |
| OOD / ablations (full) | **Not yet executed** |

Smoke numbers are pipeline checks only. Never treat unrun experiments as completed.

---

## Tests

```bash
pytest -q
```

---

## Repository layout

See the tree in the project brief: `physics/`, `data/`, `models/`, `training/`, `evaluation/`, `results/`, `paper/`, `docs/`.

---

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
