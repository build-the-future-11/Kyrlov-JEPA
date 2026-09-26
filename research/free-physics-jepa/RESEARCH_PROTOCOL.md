# Free-Physics JEPA / LMOP-JEPA — Research Protocol (Frozen)

**Freeze date:** 2026-09-26  
**Amendment:** see `amendments/PROTOCOL_AMENDMENT_0001_initial_freeze.md`

## Research question

> Does joint-embedding latent pretraining on solver-free manufactured Darcy examples transfer to genuine numerical Darcy solutions more efficiently than direct manufactured-solution regression when only a small number of genuine PDE solves are available?

## Decisive comparison

**MML-DIRECT vs LMOP-JEPA** (same manufactured corpus, matched backbone, identical genuine budgets/splits/eval).

Scratch is a secondary reference only. Beating Scratch alone does **not** support the main claim.

## PDE

\[
-\nabla\cdot\bigl(a(x)\nabla u(x)\bigr)=f(x)\quad\text{on }[0,1]^2,
\qquad u|_{\partial\Omega}=0.
\]

Discrete operator: cell-centered finite differences with **harmonic-averaged** face permeabilities; identical \(A_a\) for manufacture and genuine solves (`darcy_reference.py`).

## Grid

- Confirmatory: \(n=64\) interior DOFs per axis (\(h=1/(n+1)\)).
- Smoke: \(n=32\) (pipeline only; not scientific evidence).

## Data

**Genuine:** GRF-like log-permeability (ID correlation length \(\ell_{\mathrm{ID}}\)), fixed forcing \(f\equiv 1\), sparse solve.

**Manufactured:** \(\tilde u\) Fourier sine modes (Dirichlet-compatible); \(\tilde f=A_a\tilde u\).

- Primary family: MIXED frequency (\(1\le k,\ell\le 8\)).
- Ablation family: LOW frequency (\(1\le k,\ell\le 3\)).

**Splits:** immutable manifests + SHA256; nested \(N=25\subset 100\subset 250\); seeds \(\{11,23,47\}\).

**Primary OOD (frozen before outcomes):** permeability correlation-length shift \(\ell_{\mathrm{OOD}}<\ell_{\mathrm{ID}}\).

**Secondary OOD (exploratory only):** permeability contrast shift.

## Methods

1. Scratch — train on genuine \(N\) only.
2. MML-direct — \((a,\tilde f)\to\tilde u\) on manufactured; then fine-tune on genuine \(N\).
3. LMOP-JEPA — latent JEPA on manufactured; transfer encoder; fine-tune on genuine \(N\).

## Metrics (primary first)

1. Relative \(L_2\): \(\|û-u\|_2/\|u\|_2\)
2. \(H^1\) seminorm error
3. Energy-norm error \(\sqrt{e^\top A_a e / u^\top A_a u}\)
4. Flux error
5. Normalized PDE residual (not interchangeable with solution error)

## Label budgets

CJSJ core: \(N\in\{25,100\}\). Extension if compute allows: \(250\).

## Disallowed claims

Operator action novelty; manufactured-solution novelty; generic physics JEPA novelty; PINO/weak-form/solver-free as “firsts”; Krylov-JEPA results as LMOP evidence.

## Kill criteria

Physics fail; split leak; round-trip fail; JEPA collapse; unfair MML; test leakage; NaNs; memory crisis → stop, fix, document.
