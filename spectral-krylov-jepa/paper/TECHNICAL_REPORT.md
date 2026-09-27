---
title: "Spectral Krylov pretraining for quantum ground states"
subtitle: "A baseline audit and an unconfirmed transfer hypothesis"
date: "27 September 2026"
---

# Abstract

Short Lanczos trajectories offer inexpensive targets for learning representations
of discretized Hamiltonians. We specify a joint-embedding predictive architecture
that transfers a potential encoder to ground-state prediction. The available
16-by-16-grid, single-seed experiment does not establish a benefit from Krylov
pretraining. A new audit on the same retained test examples shows that a fixed
free-box wavefunction attains higher mean fidelity than every retained neural
method at twenty labels. Linear regression and a nine-mode Rayleigh–Ritz model
perform better still. Physics residuals expose errors that high fidelity conceals.
Fourteen supplementary variants pass a bounded execution check, but their tiny
verification runs are not efficacy evidence. The original larger experiment and
full mechanism controls remain unexecuted. The contribution at this stage is an
explicit method, reproducible failure analysis and a falsifiable evaluation plan.

# Introduction

Learning a map from a spatial potential to its lowest-energy wavefunction may
amortize the cost of repeated eigenvalue solves. Whether pretraining reduces the
number of labeled eigenpairs required is a separate empirical question. Spectral
Krylov-JEPA proposes using cheap Hamiltonian actions to construct self-supervised
targets before any labeled fine-tuning. The hypothesis is that representations
learned from these trajectories transfer better than representations learned from
potential masking or one operator action. The present evidence cannot establish
that hypothesis. We therefore report the negative baseline comparison rather than
an unsupported claim of improved sample efficiency.

# Related work

Assran et al. [1] predict image-block representations using a joint-embedding
predictive architecture. The Field-JEPA implemented here instead predicts a
full-field CLS embedding from masked potential patches; it is a domain adaptation,
not a canonical I-JEPA reproduction. Lanczos iteration and reorthogonalization are
classical numerical methods [2]. Their algebraic properties do not prove that a
neural encoder learns useful downstream representations. Physics-informed neural
quantum eigensolvers [3] already learn eigenfunctions without paired training
labels. Fourier neural operators [4] learn families of PDE solution maps. Neither
of these broader model classes was benchmarked here; superiority to them is not
claimed. The literature search and remaining novelty review are recorded in
`LITERATURE_VERIFICATION.md`.

# Problem formulation

Let the interior grid have n points on each axis of the unit square, spacing
h=1/(n+1), and zero Dirichlet boundary values. The real symmetric finite-difference
Hamiltonian is H(V) = -Delta_h/2 + diag(V), with the five-point Laplacian. Ground
truth is the smallest algebraic eigenpair (E0, psi0), computed with a sparse
symmetric eigensolver and residual rejection. Physical normalization is
h² sum_i psi_i² = 1. Lanczos vectors instead use Euclidean unit norm. These
normalizations must not be conflated. The learned map takes V and outputs an
energy and a physically normalized wavefunction, with sign ambiguity handled by
squared overlap.

# Method and exact algorithm

Choose a reproducibly seeded unit Gaussian vector q0. For j=0,...,K-1, form
w=Hq_j-beta_j q_(j-1), with the initial previous term zero; set alpha_j=q_j^T w;
subtract alpha_j q_j and fully reorthogonalize w against q0,...,q_j; then set
beta_(j+1)=||w|| and q_(j+1)=w/beta_(j+1). Breakdown triggers a recorded deterministic
restart. The corpus stores q0,...,q_K and the recurrence coefficients. No ground
state solve is needed for pretraining data.

The potential encoder E_V and state encoder E_q tokenize spatial fields with
4-by-4 patches, learned positional embeddings and transformer blocks. The default
width is 128 with four encoder blocks, four attention heads and MLP ratio four.
The historical smoke preset uses width 64, two blocks and MLP ratio two. The
Krylov predictor fuses potential tokens with k context states using type and step
embeddings and two fusion blocks. It predicts the latent of q_k from
(V,q0,...,q_(k-1)). Corpus depth K=3 and prediction context k=2 are distinct.
The target state encoder is an exponential moving average (EMA) of the online
state encoder, with no target gradient. The default EMA coefficient is 0.996 and
updates occur every optimizer step.

The pretraining loss is mean squared error between layer-normalized predicted
and target latents. An optional head predicts alpha_(k-1) and beta_k. Fine-tuning
transfers only E_V into a shared energy head and token-aware wavefunction decoder.
The frozen downstream loss combines one minus fidelity, sign-aligned mean square
wavefunction error with weight 0.1, and standardized energy mean square error
with weight 1. Potential and energy statistics come only from the selected labeled
training subset. Validation selects the checkpoint using (1-F)+0.5 relative energy
error. AdamW and cosine warmup schedules are used. Field, Operator and Krylov have
the same downstream architecture but different pretraining parameter and token
budgets. Equal step counts alone do not establish equal compute.

# Mathematical analysis

**Proposition 1 (classical span identity).** In exact arithmetic, for a real
symmetric matrix H and a nonzero q0, if no breakdown occurs through step k,
span(q0,...,q_k) equals span(q0,Hq0,...,H^k q0).

**Proof.** The base case is immediate. Assume the identity through j. The
recurrence and reorthogonalization subtract only vectors in the existing span
from Hq_j, so q_(j+1) lies in the power span through j+1. Conversely, q_j is a
polynomial of degree j in H applied to q0 whose leading coefficient is nonzero:
each nonbreakdown normalization divides by a positive beta. Thus the change of
basis between these vectors and the powers is triangular with nonzero diagonal,
and hence invertible. This proves the identity by induction. This is a statement
about the numerical target space, not a theorem about learned representations.

**Proposition 2 (fidelity does not control residual without a spectral scale).**
Let u0 and u1 be orthonormal eigenvectors of H with eigenvalues E0 and E1. For
psi=sqrt(1-epsilon)u0+sqrt(epsilon)u1, its fidelity to u0 is 1-epsilon, while
||Hpsi-E0 psi|| = sqrt(epsilon)|E1-E0|.

**Proof.** Orthonormality gives the overlap directly. Subtracting E0 psi cancels
the u0 term, leaving sqrt(epsilon)(E1-E0)u1. Taking its norm proves the claim.
A small infidelity may therefore coexist with a substantial residual, especially
on fine grids with large high-frequency eigenvalues. No transfer theorem follows.

**Proposition 3 (classical Ritz bound).** For an orthonormal basis B and symmetric
H, lambda_min(B^T H B) >= lambda_min(H). Enlarging a nested subspace cannot increase
the minimum Ritz value.

**Proof.** The Rayleigh quotient minimized over vectors Bc with ||c||=1 is minimized
over a subset of all unit vectors. Restricting the feasible set cannot reduce the
minimum. Enlarging that feasible set cannot increase it. Our one-mode and nine-mode
sine controls use nested subspaces, with numerical tests of the bound.

# Experimental setup and provenance

The historical improved smoke corpus contains 40 training, 12 validation and 12
in-distribution test examples, with 8 examples per OOD family. This actual layout
differs from the smaller smoke layout in the original protocol. All new baseline
comparisons use its saved manifest and exact stored potentials. Fine-tuning subsets
contain 10 or 20 nested labels. The historical run uses seed 11, grid 16 and 80
pretraining steps; it is a development run, not a confirmatory experiment.

Potentials are generated Gaussian mixtures, not experimental observations or
molecular data. ID mixtures have one to three negative wells with amplitudes in
[-8,-2] and widths in [0.08,0.18]. OOD families narrow the wells, deepen their
amplitudes, or enforce separated double wells. Code is under the repository MIT
license. No external dataset is redistributed. Seed ranges, manifests and hashes
support data provenance; hashes do not establish real-world applicability.

The frozen main design is n=32, 1000 unlabeled potentials, training/validation/ID
test counts 120/40/40, forty examples per OOD family, and nested label counts
10/25/50/100. Its runner retains SSL seed zero and downstream seeds 11/23/47.
That design measures downstream-seed variation conditional on one pretrained
encoder. The supplemental full control runner independently varies pretraining
and downstream seed together. Neither full campaign has executed in this work.

# Benchmarks and baseline audit

The retained neural comparisons are Scratch, Field-JEPA, Operator-JEPA and
Krylov-JEPA. The new controls use the analytic free-box ground-state shape, its
potential-specific Rayleigh energy, and a nine-mode discrete sine Rayleigh–Ritz
projection. These methods use the query Hamiltonian at inference. They are accuracy
baselines with different computational costs, not claims about matched neural
inference latency. A training-mean model and a linear ridge map from flattened
potential to aligned wavefunction and energy use only each training subset.
Ridge weight is fixed at one, with no test tuning. Predictions are normalized
before measuring fidelity.

# Measurements and analysis

Primary fidelity is |h² psi_hat^T psi0|². Energy error is
|E_hat-E0|/(|E0|+1e-6). The quantity named `residual_rel` in the code is
||H psi_hat-E_hat psi_hat||/||psi_hat||; it is normalized by wavefunction norm,
not by Hamiltonian norm, and has energy units. Residuals at the true and Rayleigh
energies separately reveal wavefunction error. All paired bootstrap comparisons
align sample IDs explicitly, with 2000 potential resamples. Their intervals are
conditional on fitted models and exclude training-seed uncertainty. Multiple
comparisons are exploratory and unadjusted; no significance-based winner is selected.

# Main results

The retained Krylov model at twenty labels has ID fidelity 0.991650 and residual
at the true energy 40.190890. Scratch at twenty labels has fidelity 0.993945 and
residual 29.043635. The new free-box baseline has fidelity 0.994460 and residual
1.483805. Nine-mode Ritz reaches fidelity 0.999898 and residual 0.761666. Linear
ridge with twenty labels reaches fidelity 0.999904 and residual 0.516766. These
results show that the observed high neural fidelities do not establish a useful
Krylov benefit on this development distribution. They do not falsify every possible
Krylov architecture or the unrun larger study.

Under the strong-amplitude shift, the retained Krylov model has fidelity 0.965578
and relative energy error 1.151349. The Ritz baseline has fidelity 0.998979 and
relative energy error 0.032930. Relative errors become sensitive when true energies
approach zero; report residuals alongside them. All audited baseline rows, including
weaker training-mean energy predictions, are retained in the generated results.

# Ablations and stability

Fourteen variants cover the baseline methods, context lengths one through three,
shuffled trajectories, removed potential tokens, a coefficient auxiliary loss,
EMA update frequency, normalized power trajectories, final latent normalization,
pretraining steps and embedding width. Their 8-grid, four-step-pretrain,
two-epoch-fine-tune execution checks completed every test split. This establishes
operational coverage only. Coefficient loss scale and width controls are specified
in `CONTROL_PROTOCOL_2026-09-27.md`. The normalized-powers control shares the K2
neural architecture and tensor dimensions, isolating orthogonalization more closely
than comparisons with a smaller field-only model. It does not equalize data-generation
cost. Removing V also prevents training the transferred potential encoder, a known
interpretation limitation. No efficacy conclusion is drawn from this tiny matrix.

# Discussion and failure modes

The development distribution is close enough to the free-box solution that high
fidelity is easy to obtain without representation learning. A smooth linear map
already exploits much of the potential dependence. The wavefunction decoder can
introduce high-frequency errors that contribute little to overlap yet dominate
physics residuals. The evidence therefore favors evaluating simple baselines and
physical error before scaling training. Changing the distribution or decoder after
seeing these outcomes would be a new exploratory study, requiring a separately
frozen confirmation protocol.

# Limitations and conclusion

The available scientific comparison is single-seed, synthetic, low resolution and
post-hoc. Pretraining collapse is not ruled out by low latent prediction loss;
representation-quality and full-scale multi-seed sensitivity remain incomplete.
No independent reproduction, production certification, quantum advantage,
foundation-model capability or state-of-the-art claim is made. Full-scale campaigns
are blocked by shared local compute contention in this execution. The project is
not submission-ready. The useful current result is a reproducible baseline failure
and a runnable mechanism study, with preserved negative findings.

# Reproducibility and AI assistance

Exact commands are in `../REPRODUCE.md`; retained receipts are at workspace root
under `astra/2026-09-27/`. Tables are generated from preserved metrics by
`scripts/13_audit_baselines.py`; control analysis uses `scripts/14_analyze_controls.py`.
The frozen source identity is saved before each new run. Local data and checkpoints
remain available but are excluded from ordinary Git storage; small receipts and
hashes are retained. The repository environment snapshot records the actual versions.
Codex (GPT-6 family; exact serving revision unavailable) assisted with code audit,
implementation, analysis and writing. The user prompt and disclosure are included
in `submission/AI_PROMPT_DISCLOSURE.md`. Human authorship, review, eligibility and
consent remain unresolved; no artificial author or affiliation is supplied.

# References

[1] M. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture," arXiv:2301.08243, 2023. https://arxiv.org/abs/2301.08243

[2] Netlib, "Full Reorthogonalization," Templates for the Solution of Algebraic
Eigenvalue Problems. https://netlib.org/utk/people/JackDongarra/etemplates/node109.html

[3] H. Jin, M. Mattheakis and P. Protopapas, "Physics-Informed Neural Networks for
Quantum Eigenvalue Problems," arXiv:2203.00451, 2022. https://arxiv.org/abs/2203.00451

[4] Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations," arXiv:2010.08895, 2020. https://arxiv.org/abs/2010.08895
