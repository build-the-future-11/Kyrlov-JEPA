# Scientific Decisions

1. **Lanczos vectors, not raw powers \(H^k q\)** — normalization and orthogonality keep trajectories well-conditioned.
2. **Fidelity as primary metric** — respects global sign ambiguity; raw MSE is secondary at best.
3. **Matched baselines** — Field/Operator/Krylov share unlabeled corpus size, steps, and encoder width where practical.
4. **Nested label subsets** — easier label-efficiency interpretation.
5. **A priori loss weights** \(\lambda_\psi=1,\lambda_E=1,\lambda_{\psi\mathrm{mse}}=0.1\) with subset-only energy/potential standardization — not tuned on test.
6. **Dual residual reporting** — residual at \(\hat E\) (protocol), at true \(E_0\) (ψ quality), and at Rayleigh \(E_R\) (eigenvector quality). High fidelity alone does not imply small residual.
7. **Shuffled and remove-V controls** — required to falsify “physics was used.”
8. **Narrow claim language** — see `paper/claims_to_evidence.md`.
