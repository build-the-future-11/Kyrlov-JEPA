# CJSJ Outline (2–3 pages)

**Working title:** Spectral Krylov-JEPA: Learning Quantum Eigenstates from Unlabeled Hamiltonian Actions

**Alternative:** Can a Model Learn Quantum Eigenstates Before Seeing Any? Krylov-Subspace Joint-Embedding Pretraining for Few-Shot Schrödinger Operators

## Research question

> Do short, inexpensive Lanczos trajectories provide a more transferable self-supervised representation for few-shot ground-state prediction than potential-field masking or a single Hamiltonian action?

## Abstract

Self-supervised pretraining on cheap operator trajectories vs expensive eigenpairs; matched baselines; few-shot fidelity/energy/residual.  
*(Results sentence: pending execution of frozen experiment.)*

## Introduction

- Few-shot eigenstate learning is label-expensive
- JEPA-style latent prediction
- Hypothesis: Krylov trajectories carry transferable Hamiltonian geometry beyond field appearance or one matvec

## Methods

- 2D FD Schrödinger operator
- Potential families (ID/OOD)
- Lanczos data generation
- Field-JEPA / Operator-JEPA / Krylov-JEPA / Scratch
- Downstream \(V\to(E_0,\psi_0)\) with sign-invariant fidelity loss

## Experimental Design

See `experiment_protocol.md` (frozen).

## Measurements/Calculations

Fidelity, relative energy error, Schrödinger residual; paired bootstrap CIs.

## Data Analysis

Mean±std over seeds; bootstrap over test potentials; nested label curves.

## Results/Discussion

**Pending execution of frozen experiment.**

Do not draft numerical claims here until `results/tables/label_efficiency.csv` exists from a real run.

## Limitations

See `limitations.md`.

## Acknowledgements

TBD.

## References

TBD (JEPA, Lanczos, neural operators / DeepONet / FNO literature, quantum ML surrogates — cite after literature pass; do not invent citations).
