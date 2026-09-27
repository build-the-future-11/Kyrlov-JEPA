# Data Generation

## Unlabeled

`generate_unlabeled_dataset` samples ID Gaussian-mixture potentials, builds \(H\), runs validated Lanczos starts, writes HDF5.

Shuffled-physics control: set `shuffle_physics=True` to pair \(V_i\) with a trajectory from \(H_{\pi(i)}\).

## Labeled

`generate_labeled_dataset` concatenates ID + OOD blocks, solves ground states with residual rejection, freezes nested split manifests under `experiments/manifests/`.

Seed ranges for families are disjoint by construction.
