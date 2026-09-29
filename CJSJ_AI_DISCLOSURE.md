# CJSJ AI assistance disclosure — consolidated submission copy

This file is the public, consolidated disclosure referenced by the CJSJ manuscript.
It preserves the retained production prompts and available tool/version information
without treating AI output as experimental evidence.

CJSJ AI PROMPT DISCLOSURE - WORKING SUBMISSION COPY
Ryan Gomez
Prepared 27 September 2026

IMPORTANT REVIEW NOTE
This file consolidates the retained major Codex project-construction prompt and the prompts used in the CJSJ manuscript-editing pass. CJSJ's posted policy asks for the full prompt used in production of the work, plus the AI tool and version, in the Methods disclosure. Before submission, compare this file against the repository's canonical submission/AI_PROMPT_DISCLOSURE.md and append any additional retained prompts or exact tool/version records that are not reproduced here.

TOOLS / VERSION INFORMATION
- ChatGPT: OpenAI, GPT-5.6 Sol (CJSJ editing and formatting pass, 27 September 2026).
- Codex: OpenAI Codex; exact serving revision is unavailable in the retained project record used to assemble this copy.

CJSJ MANUSCRIPT EDITING PROMPTS (VERBATIM)
1. make this cjsj formatted
2. is this good enough for cjsj or cound this be better
3. okay get those done then go alll in
4. okay improve it now

RETAINED MAJOR PROJECT-CONSTRUCTION PROMPT (VERBATIM)
--- BEGIN PROMPT ---
You are the lead research engineer and scientific implementation owner for a new research repository:

# Spectral Krylov-JEPA
## Krylov-Subspace Joint-Embedding Pretraining for Few-Shot Quantum Eigenstate Prediction

Your job is to take this repository from empty/new state to a scientifically rigorous, reproducible, executable research system.

Do not stop at scaffolding.

Do not leave stubs, TODOs, fake metrics, pseudocode, placeholder experiments, or unimplemented modules.

Do not invent results.

Do not claim that any experiment succeeded unless you actually ran it and the relevant logs/artifacts exist.

If compute limits prevent a full run, implement the full experiment faithfully, run the largest valid smoke/proof-of-concept configuration possible, save the real output, and clearly mark larger runs as not yet executed.

The repository must be suitable for:
1. immediate research iteration,
2. CJSJ submission preparation,
3. future expansion into a serious ML/scientific-computing paper.

---

# 0. CORE SCIENTIFIC QUESTION

We want to test:

> Can inexpensive Lanczos/Krylov trajectories provide a more useful self-supervised pretraining signal for few-shot ground-state prediction than generic field masking or a single Hamiltonian action?

The central hypothesis is that a model should learn more transferable physics by observing how a Hamiltonian acts repeatedly on states than by merely seeing the potential field.

The physical problem is the 2D stationary Schrödinger equation:

H_V ψ = E ψ

with

H_V = -1/2 ∇² + V(x,y)

on a rectangular domain with Dirichlet boundary conditions.

Input:
- potential field V(x,y)

Expensive targets:
- ground-state energy E0
- ground-state wavefunction ψ0(x,y)

Cheap pretraining supervision:
- Hamiltonian-vector products
- Lanczos vectors
- Lanczos α, β coefficients

The intended hierarchy is:

No pretraining
vs.
Field-JEPA
vs.
Operator-JEPA
vs.
Spectral Krylov-JEPA

All downstream models must use the same labeled datasets, architecture budget where possible, optimizer policy, evaluation code, and split manifests.

---

# 1. SCIENTIFIC CLAIM BOUNDARY

Do NOT make any of these claims unless experiments actually support them:

- "Krylov-JEPA is a foundation model"
- "Krylov-JEPA solves quantum mechanics"
- "Krylov-JEPA is state of the art"
- "Krylov-JEPA is globally novel"
- "Krylov-JEPA always reduces sample complexity"
- "Krylov-JEPA improves OOD generalization"

The intended claim is narrower:

> We evaluate whether self-supervised pretraining on inexpensive Lanczos/Krylov trajectories improves few-shot prediction of low-energy eigenstates relative to matched baselines.

This repository must make it easy to falsify that claim.

Negative or null results are valid and must be preserved.

---

# 2. REPOSITORY STRUCTURE

Create a clean repository approximately like:

spectral-krylov-jepa/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── physics/
│   ├── pretrain/
│   ├── finetune/
│   ├── eval/
│   └── smoke/
├── src/
│   └── spectral_krylov_jepa/
│       ├── __init__.py
│       ├── physics/
│       │   ├── grid.py
│       │   ├── potentials.py
│       │   ├── laplacian.py
│       │   ├── hamiltonian.py
│       │   ├── eigensolver.py
│       │   ├── lanczos.py
│       │   └── validation.py
│       ├── data/
│       │   ├── generate_unlabeled.py
│       │   ├── generate_labeled.py
│       │   ├── splits.py
│       │   ├── datasets.py
│       │   ├── storage.py
│       │   └── manifests.py
│       ├── models/
│       │   ├── encoders.py
│       │   ├── patch_embed.py
│       │   ├── field_jepa.py
│       │   ├── operator_jepa.py
│       │   ├── krylov_jepa.py
│       │   ├── target_encoder.py
│       │   ├── predictors.py
│       │   ├── downstream.py
│       │   └── heads.py
│       ├── training/
│       │   ├── pretrain.py
│       │   ├── finetune.py
│       │   ├── ema.py
│       │   ├── losses.py
│       │   ├── optim.py
│       │   ├── checkpointing.py
│       │   └── seed.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── residual.py
│       │   ├── bootstrap.py
│       │   ├── ood.py
│       │   ├── ablations.py
│       │   └── evaluate.py
│       ├── plotting/
│       │   ├── physics_figures.py
│       │   ├── learning_curves.py
│       │   ├── ood_figures.py
│       │   └── ablation_figures.py
│       └── utils/
│           ├── io.py
│           ├── logging.py
│           ├── config.py
│           └── device.py
├── scripts/
│   ├── 00_validate_physics.py
│   ├── 01_generate_unlabeled.py
│   ├── 02_generate_labeled.py
│   ├── 03_pretrain_field_jepa.py
│   ├── 04_pretrain_operator_jepa.py
│   ├── 05_pretrain_krylov_jepa.py
│   ├── 06_finetune.py
│   ├── 07_evaluate.py
│   ├── 08_run_ablation.py
│   ├── 09_generate_figures.py
│   └── 10_run_smoke_suite.py
├── tests/
│   ├── test_grid.py
│   ├── test_laplacian.py
│   ├── test_hamiltonian.py
│   ├── test_eigensolver.py
│   ├── test_lanczos.py
│   ├── test_potentials.py
│   ├── test_datasets.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_end_to_end_smoke.py
├── experiments/
│   ├── manifests/
│   ├── raw/
│   ├── logs/
│   └── summaries/
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── bootstrap/
│   └── tables/
├── figures/
├── paper/
│   ├── claims_to_evidence.md
│   ├── experiment_protocol.md
│   ├── cjsj_outline.md
│   ├── limitations.md
│   └── results_status.md
└── docs/
    ├── architecture.md
    ├── data_generation.md
    ├── reproducibility.md
    └── scientific_decisions.md

Use sensible deviations if needed, but preserve this separation of:
- physics
- data
- models
- training
- evaluation
- results
- documentation

---

# 3. PHYSICS ENGINE

Implement the physical problem correctly before any ML.

## Domain

Start with:

x,y ∈ [0,1]

Default interior grid:
32 × 32

Use configurable resolution.

Dirichlet boundary condition:

ψ = 0 on ∂Ω

Use a sparse finite-difference discretization.

## Laplacian

Implement a verified 2D finite-difference Laplacian with correct grid spacing.

The Hamiltonian is:

H = -1/2 Δ + diag(V)

Use scipy.sparse where appropriate.

The matrix must be real symmetric/Hermitian to numerical tolerance.

## Validation

Build tests that verify:
- symmetry
- shape
- correct number of degrees of freedom
- finite eigenvalues
- known analytic or semi-analytic sanity cases where possible

Include at least one simple potential:
- infinite-box-like V = 0 on the interior with zero boundary

Check that qualitative low-energy eigenmodes are sensible.

---

# 4. POTENTIAL GENERATOR

Implement reproducible random potential families.

## ID family

Smooth Gaussian mixtures:

V(x,y) = sum_i A_i exp(-((x-x_i)^2 + (y-y_i)^2)/(2 σ_i²))

with configurable:
- 1–3 wells
- center positions
- amplitudes
- widths

Use deterministic seedable generation.

## OOD families

Implement at least:

1. narrow wells
   - σ below training range

2. strong-amplitude wells
   - amplitudes outside training range

3. separated double wells
   - two distinct minima with larger spatial separation

4. optionally rough/high-frequency perturbations
   - only if numerically stable and clearly separated from the ID generator

Every dataset example must store:
- generator family
- seed
- parameters
- grid resolution
- domain specification

---

# 5. EIGENSOLVER

Use a sparse symmetric eigensolver such as scipy.sparse.linalg.eigsh.

Compute at least:
- lowest eigenvalue E0
- corresponding eigenvector ψ0

Normalize ψ0.

Handle sign ambiguity in evaluation.

Verify:

||Hψ0 - E0 ψ0||

is small.

Store:
- potential
- E0
- ψ0
- residual norm
- exact config/hash where useful

Do not silently accept bad eigensolutions.

Set tolerances.

Reject or flag examples that fail validation.

---

# 6. LANCZOS / KRYLOV GENERATOR

Implement numerically stable Lanczos iteration.

Given normalized q0:

β0 = 0

for j:

w = H qj - βj q_{j-1}
αj = qj^T w
w = w - αj qj

Use reorthogonalization if required for short trajectories.

β_{j+1} = ||w||

q_{j+1} = w / β_{j+1}

Default Krylov depth:
K = 3 or 4

Store:
- q0
- q1
- q2
- q3
- α values
- β values

Validate:
- ||qi|| ≈ 1
- qi^T qj ≈ 0 for i != j
- recurrence residual is small
- no NaNs
- β values valid

If breakdown occurs, regenerate q0 and record it.

Do NOT generate raw powers H^k q without normalization as the primary representation.

Lanczos vectors are the canonical data.

---

# 7. UNLABELED PRETRAINING DATA

Generate a corpus using potentials only, with no eigenstate computation.

Prototype size:
1,000 potentials

Serious run target:
5,000–10,000 potentials

For each V:
- construct H
- generate 1–4 random starts q0
- compute short Lanczos sequence

Store data efficiently.

Preferred storage:
- HDF5, Zarr, NPZ shards, or PyTorch-safe format
- choose based on simplicity and reliability

Never serialize huge sparse matrices per example if reconstructing H from V is cheaper.

Store manifest metadata.

---

# 8. LABELED DATA

Create separate labeled data with actual eigensolves.

Initial:
200–500 solved potentials

Suggested split:

train_pool
validation
test_ID
test_OOD_narrow
test_OOD_strong
test_OOD_double

Freeze manifests before serious runs.

Never modify the test set based on observed outcomes.

Never allow potentials or near-identical generator seeds to leak across splits.

Fine-tuning subsets:

N = 10
N = 25
N = 50
N = 100

Optionally 250 if data exists.

Subsets must be nested where practical:
10 ⊂ 25 ⊂ 50 ⊂ 100

so label-efficiency curves are easier to interpret.

---

# 9. MODEL FAMILY

Use a small architecture.

Do not build a giant transformer.

Default target:
1–5 million parameters.

Make architecture size configurable.

## Shared components

Potential encoder E_V(V)

State encoder E_q(q)

Target encoder E_T, EMA-updated from corresponding online encoder where JEPA requires it.

Predictor P.

Patch embedding:
4×4 patches for 32×32 inputs is a reasonable start.

Default:
- embedding dim 128
- 4 transformer blocks
- 4 heads
- modest MLP ratio

Do not assume this is optimal.
This is a controlled proof-of-concept.

---

# 10. BASELINE A — SCRATCH

No pretraining.

Train:

V → (E0, ψ0)

using identical downstream architecture.

Downstream output:
- scalar energy head
- spatial wavefunction decoder/head

Normalize predicted wavefunction before fidelity evaluation.

Loss may include:
- sign-invariant wavefunction reconstruction
- energy MSE

Use carefully chosen weights.

Document them.

Do not use test data for loss tuning.

---

# 11. BASELINE B — FIELD-JEPA

This is the generic self-supervised baseline.

Use only V.

Mask spatial patches of V.

Online/context encoder receives visible V patches.

Target encoder receives hidden/full V representation.

Predict latent target embeddings.

This approximates the philosophy of field-structure pretraining.

Match:
- parameter count
- pretraining steps
- batch size
- optimizer family
- unlabeled examples

as closely as practical to Krylov-JEPA.

---

# 12. BASELINE C — OPERATOR-JEPA

Use one Hamiltonian action.

Input:
V, q

Target:
latent representation of Hq

or normalized one-step transformed state.

Preferred task:

(V, q) → z(normalize(Hq))

with target encoder.

This tests whether learning one operator action is sufficient.

Again match compute and architecture as closely as possible.

---

# 13. MAIN MODEL — SPECTRAL KRYLOV-JEPA

Use actual Lanczos trajectories.

Main task:

(V, q0, q1) → z(q2)

Optionally support deeper:

(V, q0, q1, q2) → z(q3)

The potential encoder must participate materially.

Do not allow the predictor to solve the task while ignoring V.

Include an ablation/control to test this.

Possible architecture:
- encode V separately
- encode each qj
- fuse tokens through cross-attention or concatenated transformer tokens
- predictor outputs latent representation corresponding to next Lanczos state

Target encoder:
EMA copy

Main JEPA loss:

L_K = mean || z_pred - stopgrad(z_target) ||²

Prevent collapse using appropriate stable JEPA practice:
- EMA target encoder
- normalization
- optional variance/covariance regularization only if needed
- document every addition

Do not hide instability.

---

# 14. OPTIONAL AUXILIARY LANCZOS LOSS

Lanczos coefficients are physically meaningful and cheap.

Optionally predict:

αj
βj+1

from the context.

Auxiliary loss:

L_coeff =
MSE(α_hat, α)
+
MSE(β_hat, β)

Then:

L_total =
L_K
+
λ_coeff L_coeff

This must be optional/configurable.

Run an ablation:
- JEPA only
- JEPA + coefficient prediction

Do not assume the auxiliary task helps.

---

# 15. DOWNSTREAM FINE-TUNING

After pretraining, transfer the potential encoder into the downstream model.

Use the same downstream head and training code for:

- Scratch
- Field-JEPA
- Operator-JEPA
- Krylov-JEPA

Fine-tune using labeled subsets:
10
25
50
100

Use:
3 seeds where feasible:
11
23
47

Freeze or fine-tune encoder according to a predeclared default.

Preferred main setting:
fine-tune the full encoder.

Optional secondary:
linear/partial probe.

Do not choose whichever looks better after seeing test results.

---

# 16. DOWNSTREAM LOSSES

Ground state has global sign ambiguity.

Use sign-invariant wavefunction loss.

For normalized ψ and ψ_hat:

fidelity:

F = |<ψ_hat, ψ>|²

Potential training loss options:
- 1 - F
- sign-aligned MSE
- energy MSE

Preferred combined objective:

L =
λ_ψ (1 - F)
+
λ_E normalized_energy_MSE

Do not include test residual in training unless explicitly running a physics-informed downstream ablation.

The primary experiment should isolate representation pretraining.

---

# 17. METRICS

Primary metric:

## Ground-state fidelity

F = |<ψ_hat, ψ>|²

for normalized states.

Report:
1 - F as error if useful.

Secondary metrics:

## Relative energy error

|E_hat - E0| / (|E0| + eps)

## Schrödinger residual

r = H ψ_hat - E_hat ψ_hat

report:

||r||2 / ||ψ_hat||2

## Optional
- sign-aligned relative L2 error
- residual-energy relationship

Do not report raw MSE alone as the main wavefunction metric.

---

# 18. OOD EVALUATION

Evaluate every final model on:

1. ID Gaussian wells
2. narrower wells
3. stronger amplitudes
4. separated double wells

Do not retune on OOD data.

Report:
- fidelity
- relative energy error
- residual

Test the hypothesis that operator-aware pretraining may degrade less under structural shift.

Do not phrase this as true unless supported.

---

# 19. ABLATIONS

Implement these from the start.

## A. Krylov depth

K1:
(V, q0) → q1 representation

K2:
(V, q0, q1) → q2 representation

K3:
(V, q0, q1, q2) → q3 representation

Compare downstream label efficiency.

Question:
Does performance saturate with Krylov depth?

## B. Shuffled physics control

Break physical consistency:

pair V_A with a trajectory generated under V_B.

Train using the same amount of data.

If it performs similarly, the method may be exploiting generic structure instead of Hamiltonian-specific dynamics.

This is a critical negative control.

## C. Remove V

Train:

(q0, q1) → q2

without the potential.

Question:
Does the model actually need Hamiltonian information?

## D. Single-action vs trajectory

Operator-JEPA vs Krylov-JEPA.

## E. Coefficient prediction

with vs without α/β auxiliary loss.

---

# 20. EXPERIMENTAL PROTOCOL

Before serious result runs, create:

paper/experiment_protocol.md

containing frozen decisions for:
- grid
- potential ranges
- dataset sizes
- split seeds
- model architecture
- optimizer
- learning rates
- batch size
- pretraining steps
- fine-tuning epochs
- early stopping policy
- metrics
- OOD definitions
- ablations

After this protocol is frozen, any change based on test results must be labeled exploratory.

Do not silently rewrite the protocol.

---

# 21. STATISTICS

For each final method and label count:

run multiple seeds where feasible.

Report:
- mean
- standard deviation
- all seed values

For final test comparisons:
use paired bootstrap over test potentials.

Preferred:
2,000 bootstrap resamples

Report:
95% confidence intervals.

Implement bootstrap utilities.

Do not pretend training-seed uncertainty and finite-test-set uncertainty are the same thing.

---

# 22. SANITY CHECKS BEFORE ML

Create script:

scripts/00_validate_physics.py

It must:

1. generate one potential
2. construct H
3. verify symmetry
4. compute E0, ψ0
5. verify normalization
6. compute residual
7. generate Lanczos sequence
8. verify orthogonality
9. verify recurrence
10. generate figure containing:
   - potential
   - ground-state ψ0
   - |ψ0|²
   - residual map

Save figure.

Fail loudly if tolerances are violated.

No ML should run before this passes.

---

# 23. SMOKE EXPERIMENT

Create a tiny but complete proof-of-concept pipeline.

Suggested smoke dataset:

unlabeled:
100–300 potentials

labeled:
40–80 solved potentials

fine-tuning:
10 or 20 labels

tiny model:
reduced embedding/layers

Train:
- scratch
- Krylov-JEPA

Run real evaluation.

Save real metrics.

Purpose:
verify the full pipeline, not prove the hypothesis.

Command should be approximately:

python scripts/10_run_smoke_suite.py

It must execute end-to-end.

---

# 24. FIRST DECISIVE EXPERIMENT

Once smoke passes, support this exact experiment:

Unlabeled:
1,000–5,000 potentials

Labeled:
at least 200 total

Compare:
- Scratch
- Field-JEPA
- Operator-JEPA
- Krylov-JEPA

Label budgets:
10
25
50
100

Seeds:
at least 3 if compute allows

Metrics:
- fidelity
- energy error
- residual

Main output:
results/tables/label_efficiency.csv

and figure:

figures/label_efficiency.png

x-axis:
number of solved Hamiltonians

y-axis:
ground-state fidelity or infidelity

The graph must show all methods.

---

# 25. VISUALIZATION

Generate clean scientific figures with matplotlib.

Do not use decorative styling.

Create:

## Figure A
Potential | Ground state | probability density | residual

## Figure B
Label efficiency curve

Scratch
Field-JEPA
Operator-JEPA
Krylov-JEPA

## Figure C
ID vs OOD grouped comparison

## Figure D
Krylov-depth ablation

## Figure E
Example predictions:
true ψ0
predicted ψ0
absolute error

All figures must use real results.

No fabricated charts.

---

# 26. RESULT ARTIFACTS

Every run must create a unique run directory.

Store:
- config snapshot
- git commit hash if available
- seed
- device
- package versions
- stdout/log
- best checkpoint
- predictions
- metrics
- timing
- failure status

Example:

experiments/raw/<run_id>/

Use deterministic naming.

---

# 27. REPRODUCIBILITY

Implement global seeding for:
- Python
- NumPy
- PyTorch

Set deterministic options where practical.

Record:
- OS
- Python
- PyTorch
- scipy
- numpy
- device
- CPU/GPU
- CUDA/MPS if applicable

Support:
- CPU
- CUDA if available
- Apple MPS if stable

Default safely based on detected backend.

Do not crash just because CUDA is absent.

---

# 28. MEMORY / LOCAL MACHINE SAFETY

The system may run on a 16 GB MacBook.

Default configs must be conservative.

Avoid:
- loading all datasets into RAM
- huge dense Hamiltonian matrices
- excessive dataloader workers
- giant transformers

Use:
- sparse physics matrices
- memory-mapped/sharded datasets if needed
- small batch sizes
- config options for workers

Target routine memory use well under 12 GB.

Heavy runs should be configurable for cloud GPU separately.

---

# 29. TEST SUITE

Write meaningful pytest tests.

Minimum tests:

## Physics
- Laplacian symmetry
- Hamiltonian symmetry
- correct shapes
- simple eigenpair residual
- Lanczos orthogonality
- Lanczos recurrence
- reproducible potential generation

## Data
- no overlap among frozen split manifests
- correct shapes/dtypes
- deterministic seeded generation

## Models
- forward shape
- finite loss
- EMA update
- no NaNs
- gradients exist on online encoder
- target encoder does not receive gradients

## Metrics
- fidelity invariant under ψ → -ψ
- exact prediction gives fidelity ≈ 1
- residual small for true eigenpair

## End-to-end
tiny generation → pretraining → fine-tuning → evaluation

All tests must pass.

---

# 30. README

Write a strong README including:

## Overview
What the scientific question is.

## Core idea
Cheap:
Hq / Lanczos actions

Expensive:
eigensolves

## Research comparison
Scratch vs Field-JEPA vs Operator-JEPA vs Krylov-JEPA

## Installation

## Quickstart

## Physics validation

## Generate data

## Pretrain

## Fine-tune

## Evaluate

## Reproduce smoke run

## Reproduce main experiment

## Results status
Explicitly distinguish:
- implemented
- smoke-tested
- full results not yet run
- completed experimental results

Never imply unrun results.

---

# 31. SCIENTIFIC DOCUMENTATION

Create:

paper/claims_to_evidence.md

with rows such as:

Claim:
Krylov pretraining improves label efficiency.

Required evidence:
matched downstream model, fixed label subsets, multiple seeds, held-out test.

Status:
not tested / mixed / supported / unsupported.

Also create:

paper/results_status.md

with explicit:
- completed runs
- incomplete runs
- failed runs
- no-results-yet warnings

And:

paper/limitations.md

Include:
- synthetic potential family
- finite-difference discretization
- small grids
- only ground state
- possible mismatch between pretraining trajectory task and eigensolution task
- model-scale limitations
- no claim of quantum advantage
- no claim of universal PDE transfer

---

# 32. CJSJ PAPER SUPPORT

Create:

paper/cjsj_outline.md

Target a concise 2–3 page paper.

Suggested title:

"Spectral Krylov-JEPA: Learning Quantum Eigenstates from Unlabeled Hamiltonian Actions"

Alternative:

"Can a Model Learn Quantum Eigenstates Before Seeing Any? Krylov-Subspace Joint-Embedding Pretraining for Few-Shot Schrödinger Operators"

Sections:

Abstract
Introduction
Methods
Experimental Design
Measurements/Calculations
Data Analysis
Results/Discussion
Limitations
Acknowledgements
References

Do NOT invent the Results section.

Instead insert:
"Pending execution of frozen experiment."

Also include the exact research question:

> Do short, inexpensive Lanczos trajectories provide a more transferable self-supervised representation for few-shot ground-state prediction than potential-field masking or a single Hamiltonian action?

---

# 33. EXPERIMENT PRIORITY ORDER

Do the work in this exact priority:

P0
Physics correctness

P1
Lanczos correctness

P2
Dataset generation

P3
End-to-end smoke experiment

P4
Scratch vs Krylov-JEPA proof of concept

P5
Field-JEPA and Operator-JEPA baselines

P6
Full label-efficiency experiment

P7
OOD tests

P8
Krylov-depth and shuffled controls

P9
Figures + tables

P10
Paper-support docs

Do not waste time on polish before P0–P4 work.

---

# 34. STOP / KILL CRITERIA

Be scientifically ruthless.

If any of these happen, document them:

- Krylov-JEPA fails to beat scratch consistently at low label count
- Field-JEPA matches/exceeds Krylov-JEPA
- Operator-JEPA matches/exceeds Krylov-JEPA
- shuffled-Krylov performs equally well
- removing V has no effect
- Lanczos pretraining collapses
- improvements disappear under seeds
- fidelity improves but residual worsens badly
- OOD performance collapses

Do not hide these.

They may become the actual scientific result.

---

# 35. CODE QUALITY

Use:
- type hints
- docstrings
- structured configs
- robust errors
- logging
- tests

Avoid:
- notebooks as the main pipeline
- giant monolithic scripts
- global magic constants
- hardcoded absolute paths
- silent exception swallowing
- hidden test leakage

A notebook may be added only for optional exploration/visualization.

Canonical runs must be CLI/script based.

---

# 36. IMMEDIATE EXECUTION REQUIREMENT

After implementing the repository:

1. install dependencies if possible
2. run pytest
3. run physics validation
4. run the end-to-end smoke suite
5. inspect failures
6. repair them
7. rerun until the smoke pipeline is actually functional
8. save real logs/results
9. generate at least one real physics figure
10. generate a real smoke metrics table

Do not stop after "code written."

The acceptance criterion is:

> A fresh user can clone the repository, run one documented command, and reproduce a valid tiny Krylov-JEPA experiment from data generation through evaluation.

---

# 37. FINAL RESPONSE

When finished, report:

## Implemented
specific modules/features.

## Verified
tests and commands actually run.

## Real smoke results
only numbers actually produced.

## Not yet executed
larger experiments that remain.

## Scientific risks found
anything that may undermine the idea.

## Exact next command
the next best command for running the larger experiment.

## Repository status
whether it is:
- scaffolded
- runnable
- smoke-verified
- experiment-ready
- full-results-ready

Never report an unexecuted experiment as complete.

---

# FINAL RESEARCH PRINCIPLE

The project is testing a single idea:

Generic JEPA says:

"learn what the system looks like."

Operator-JEPA says:

"learn what the operator does once."

Spectral Krylov-JEPA says:

"learn how the operator moves states through a physically meaningful subspace."

The entire codebase and experimental design should make it possible to determine whether that extra information actually reduces the number of expensive quantum eigensolutions required downstream.

Build the simplest system capable of answering that question rigorously.
--- END PROMPT ---
