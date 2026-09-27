# Spectral Krylov pretraining for quantum ground states

## Abstract

Predicting quantum ground states from a spatial potential requires learning both a smooth wavefunction and a physically consistent energy. We examine whether short Lanczos trajectories can pretrain a transferable representation before labeled eigenpairs are available. Our joint-embedding predictive architecture learns to predict the next trajectory state's representation from the potential and preceding states. An audit of the retained small-scale experiment finds no established benefit from this pretraining. On twelve held-out potentials, a fixed free-box wavefunction has mean fidelity 0.994460, compared with 0.991650 for Krylov pretraining followed by twenty-label fine-tuning. A nine-mode classical subspace model reaches 0.999898 and substantially smaller physics residuals. These results expose the difficulty of interpreting high overlap on a simple distribution. The method and additional controls are executable, but the larger confirmatory experiment has not run. We report the bounded negative evidence and distinguish it from the still-open transfer hypothesis.

## Introduction

A quantum ground state is the lowest-energy stationary solution of a Hamiltonian, the operator that describes a system's kinetic and potential energy. Repeatedly solving for that state can be expensive when many potentials must be studied. A learned surrogate could predict solutions after seeing relatively few labeled examples. This motivates asking whether inexpensive operations with the Hamiltonian can provide useful pretraining signals.

Joint-embedding predictive architectures learn by predicting representations rather than reconstructing raw observations [1]. Lanczos iteration constructs an orthogonal basis from repeated matrix-vector products [2]. Spectral Krylov-JEPA combines these ideas: it predicts the latent representation of a later Lanczos vector, then transfers the potential encoder to ground-state regression. The proposed contribution concerns transfer from operator trajectories. Neural eigenvalue solvers and operator learning already exist [3,4], and this work does not claim to introduce either field.

The falsifiable hypothesis is that Krylov pretraining improves label efficiency relative to scratch training, potential masking and a single Hamiltonian action. Accurate downstream overlap alone is insufficient evidence: a simple potential distribution may already be well approximated by a fixed wavefunction. We therefore audit the retained predictions against simple classical and statistical baselines before interpreting the neural results.

## Methods

We discretize the stationary Schrodinger equation on the unit square with zero wavefunction at the boundary. With n interior points per axis and spacing h=1/(n+1), the real symmetric Hamiltonian is H(V) = -Delta_h/2 + diag(V), where Delta_h is the five-point finite-difference Laplacian and V is the potential. Ground-truth eigenpairs come from a sparse symmetric eigensolver, with residual rejection. Wavefunctions satisfy h squared times their squared Euclidean norm equal to one. Lanczos vectors instead have unit Euclidean norm.

Starting from a seeded random vector q0, Lanczos repeatedly applies H, subtracts the current and previous basis components, and reorthogonalizes against all earlier vectors. Dividing the remainder by its norm gives the next vector. In exact arithmetic without breakdown, the first k+1 vectors span the same space as q0, Hq0, through H to the power k applied to q0. This classical span identity follows because each new vector is a degree-k polynomial in H with nonzero leading coefficient. It does not prove that a neural network learns transferable features.

Potential and state encoders use spatial patches, positional embeddings and transformer blocks. The default prediction takes V, q0 and q1 as context and predicts the representation of q2. The target encoder is an exponential moving average of the online state encoder and receives no gradients. The pretraining objective is mean squared error between normalized predicted and target representations. Only the potential encoder transfers to a common downstream energy head and wavefunction decoder. Normalization statistics use the labeled training subset, and validation selects the checkpoint. The full technical report gives the recurrence, losses, architecture and implementation details.

## Experimental design

The retained development experiment uses a 16-by-16 interior grid, seed 11 and eighty pretraining steps. Its saved manifest contains forty training potentials, twelve validation potentials, twelve in-distribution test potentials and eight examples in each of three distribution shifts. The nested labeled subsets contain ten or twenty examples. This actual run differs from the smaller smoke layout described in the original planning document; the saved run artifacts govern this audit.

All potentials are generated Gaussian mixtures. The in-distribution family contains one to three negative wells. The shifted families have narrower wells, deeper wells or separated double wells. These are synthetic numerical examples, with no human participants or experimental molecular observations. The repository's code and deterministic generation procedure provide provenance. No external dataset was used.

The neural methods are scratch training, Field-JEPA, Operator-JEPA and Krylov-JEPA. Field-JEPA predicts a full-field representation from masked potential patches, making it an adaptation rather than a canonical reproduction of image I-JEPA. Operator-JEPA uses one normalized Hamiltonian action. The downstream model is shared, but pretraining token and parameter counts differ, so equal steps do not establish equal compute.

We add a fixed free-box sine wave with a potential-specific Rayleigh energy, a nine-mode sine Rayleigh-Ritz projection, the mean training wavefunction and energy, and linear ridge regression from potential to wavefunction and energy. Ridge weight is fixed at one. Signs and regression statistics are determined using training examples only. The Ritz baseline accesses the query Hamiltonian at inference and therefore has a different cost profile from a neural surrogate.

## Measurements and data analysis

Fidelity is the squared overlap between normalized predicted and reference wavefunctions and is unchanged by a global sign reversal. We also measure relative energy error and the norm of H times the predicted wavefunction minus energy times that wavefunction, divided by the wavefunction norm. This residual has energy units and is not divided by the Hamiltonian norm. Residuals at the true energy isolate wavefunction quality; Rayleigh-energy residuals supply a complementary physical check.

High fidelity need not imply a small residual. For an orthonormal ground state u0 and excited state u1, a normalized mixture with excited-state squared amplitude epsilon has fidelity 1-epsilon. Its residual at the true ground energy is the square root of epsilon times the eigenvalue gap. Even a small admixture can therefore produce a large residual when the gap is large.

Baseline comparisons align the exact retained test example indices. Paired bootstrap intervals use two thousand resamples of potentials and are conditional on the fitted model. They exclude training-seed uncertainty. The audit is exploratory and its multiple intervals are unadjusted. No method, test subset or reported seed is selected to obtain a favorable outcome. All per-example metrics and baseline rows remain available.

## Results and discussion

At twenty labels, Krylov-JEPA has mean in-distribution fidelity 0.991650 and residual at the true energy 40.190890. Scratch reaches 0.993945 with residual 29.043635. The fixed free-box baseline reaches 0.994460 with residual 1.483805. Nine-mode Ritz reaches 0.999898 with residual 0.761666, while twenty-label linear ridge reaches 0.999904 with residual 0.516766. Figure 1 summarizes the simple baseline infidelities. Their strong performance indicates that this development distribution supplies a weak test of the proposed representation-learning advantage.

The paired Krylov-minus-free-box fidelity difference is -0.002810, with a conditional 95 percent bootstrap interval from -0.006733 to 0.000182. Thus the mean ordering alone is not a decisive statistical separation. Against nine-mode Ritz, the difference is -0.008249 with interval from -0.012187 to -0.005594. This is a bounded comparison on the historical development corpus, not a statement about all possible Krylov models.

Under the strong-amplitude shift, the retained Krylov model has fidelity 0.965578 and relative energy error 1.151349. Nine-mode Ritz reaches 0.998979 and 0.032930 respectively. Relative energy errors can become large near zero true energy, reinforcing the need for residuals and per-example evidence. The smooth fixed or linear solutions also avoid much of the high-frequency error present in the neural decoder outputs.

Fourteen supplementary variants cover context order, shuffled physics, removal of the potential, coefficient loss, target update frequency, subspace construction, latent normalization, step budget and representation dimension. All completed a tiny execution check. Those runs verify implementation paths and cannot establish convergence or accuracy benefits. The normalized-power control uses the same neural architecture as the default Krylov model, providing a closer parameter and operation-shape match. Removing V leaves the transferred potential encoder untrained, so that variant is a transfer-path sanity check with a known interpretation limitation.

## Limitations and conclusion

The present evidence is synthetic, low resolution, single-seed and post-hoc. Low pretraining loss does not rule out representation collapse. The frozen larger experiment uses a 32-by-32 grid, one thousand unlabeled potentials, four label counts and three downstream seeds. It has not run, and those seeds would still share one pretrained encoder. The full supplemental design adds independent pretraining seeds, but also remains unexecuted. Neither full-scale efficacy nor independent reproduction is established.

These findings motivate a narrower conclusion: the existing high fidelities do not demonstrate useful Krylov transfer, and simple baselines expose a substantial physical-accuracy deficit. They do not establish that the broader hypothesis is false. Changing the data distribution or decoder after this audit would require a new, clearly labeled protocol. The retained negative result, exact method and executable controls support that future test without treating an implementation check as scientific confirmation.

## Acknowledgements and reproducibility

Codex, using the GPT-6 model family with exact serving revision unavailable, assisted with code audit, implementation, analysis and writing. The full task prompt and disclosure are supplied in AI_PROMPT_DISCLOSURE.md. Methods, artifacts and reproduction commands accompany the technical report. Human author information, eligibility, consent and final scientific review remain to be completed before submission. No author, affiliation, funding or approval has been invented.

## References

[1] M. Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, arXiv:2301.08243, 2023.

[2] Netlib, Full Reorthogonalization, Templates for the Solution of Algebraic Eigenvalue Problems, netlib.org/utk/people/JackDongarra/etemplates/node109.html.

[3] H. Jin, M. Mattheakis and P. Protopapas, Physics-Informed Neural Networks for Quantum Eigenvalue Problems, arXiv:2203.00451, 2022.

[4] Z. Li et al., Fourier Neural Operator for Parametric Partial Differential Equations, arXiv:2010.08895, 2020.
