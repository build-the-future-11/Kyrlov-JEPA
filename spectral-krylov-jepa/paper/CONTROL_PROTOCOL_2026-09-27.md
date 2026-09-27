# Supplementary controls — frozen before the new execution

This supplement preserves `experiment_protocol.md` unchanged. The historical smoke
outcomes were already known. New controls are exploratory relative to that history;
verification-scale outputs are engineering evidence only. No efficacy selection or
publication gate is based on their two-example splits.

The full control runner uses the original 32-grid data sizes, split seed, label
subsets, optimizers and losses, with independent SSL and downstream seeds 11/23/47.
The original main runner retains its single SSL seed 0, so its three downstream
seeds are conditional on one pretrained encoder. Do not call them independent SSL
replications. Repeated test access across these studies must be disclosed.

`12_run_controls.py` fixes fourteen variants: scratch, field, operator, K1, K2, K3,
cyclic shuffled trajectory donors, remove-V, coefficient loss weight 1e-4, EMA update
every fourth step, normalized powers in place of Lanczos, no final latent
normalization, half pretrain steps, and half embedding width. Width changes only
embedding dimension (heads/depth/MLP ratio remain fixed). The coefficient scale is
an exploratory choice, not tuned on outcomes. Coefficient loss uses raw coefficients;
large finite-difference scales can dominate and must be reported as a failure mode.

The powers control uses the identical K2 architecture, parameter count, batch and
step count, with q[j+1]=Hq[j]/||Hq[j]||. It removes Lanczos orthogonalization while
retaining Hamiltonian actions. This matches neural training tensor shapes and steps,
not total data-generation cost. Report elapsed time and parameter counts; do not
claim strict measured FLOP or wall-clock equality. Operator and Field are adapted
baselines, not canonical I-JEPA reproductions. Remove-V does not train the transferred
potential encoder; it is therefore a transfer-path sanity check, not a clean estimate
of the incremental information in V.

New no-label baseline audit: use one free-box sine mode and 3x3 sine-mode
Rayleigh–Ritz projection on every stored test potential. Also report a mean training
wavefunction/energy and linear ridge regression using each training subset, fixed
ridge 1.0, without test selection. These classical baselines have different inference
costs. Do not interpret high fidelity alone as accurate physics.

Compare paired per-potential fidelity differences with 2000 bootstrap resamples.
Seed-conditional intervals do not include training-seed uncertainty. Full-scale
analysis must separately show all seeds, missing/failed cells, and variability.
The primary hypothesis stays unestablished until the original matrix and controls
are complete. No selecting the best ablation after test inspection.

Run identities bind configuration and source hashes; data hashes bind the corpus.
A resume with any mismatch must fail. Never delete another process's lock. All
full runs remain sequential and require an available local compute slot.
