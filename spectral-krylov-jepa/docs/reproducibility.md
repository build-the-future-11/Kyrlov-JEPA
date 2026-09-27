# Reproducibility

- `seed_everything` seeds Python, NumPy, PyTorch; enables deterministic flags where practical (`warn_only` to avoid hard crashes on MPS).
- Each run directory stores `config_snapshot.json`, `metrics.json`, `status.json`, environment versions, and git hash when available.
- Device selection: `auto` → CUDA → MPS → CPU.
- Default dataloader `num_workers=0` for MacBook memory safety.
- Prefer reconstructing \(H\) from \(V\) over storing dense/sparse matrices.
