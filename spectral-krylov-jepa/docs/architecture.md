# Architecture

## Physics

`GridSpec` → sparse Laplacian → `H = -½Δ + diag(V)` → `eigsh` ground state / short Lanczos.

## Data

HDF5 stores potentials and (for unlabeled) Lanczos vectors \(q_0..q_K\), \(\alpha\), \(\beta\). Sparse \(H\) is never serialized; it is reconstructed from \(V\).

## Models

Shared potential encoder \(E_V\) (patch embed + small transformer).

- **Field-JEPA:** mask patches of \(V\); predict EMA target CLS
- **Operator-JEPA:** fuse \((V,q)\); predict EMA encoding of \(\mathrm{normalize}(Hq)\)
- **Krylov-JEPA:** fuse \((V,q_0,\ldots,q_{k-1})\); predict EMA encoding of \(q_k\)
- **Downstream:** \(E_V(V)\) → energy head + wavefunction decoder

EMA target encoders never receive gradients.

## Training

`training/pretrain.py` and `training/finetune.py` write run dirs under `experiments/raw/` with config snapshots, logs, checkpoints, and metrics.
