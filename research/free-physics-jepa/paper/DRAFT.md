# LMOP-JEPA / Free-Physics JEPA — paper draft

## Title

Does latent predictive pretraining on manufactured Darcy examples transfer better than direct manufactured-solution regression under small genuine-label budgets? An audited negative-methods report.

## Abstract

We set out to test a narrow comparative transfer hypothesis on 64×64 Darcy flow: whether LMOP-JEPA (joint-embedding latent pretraining on solver-free manufactured solutions) improves few-shot accuracy on genuine numerical solutions relative to MML-direct (direct manufactured-solution regression), with matched corpora, matched FNO backbones, budgets \(N\in\{25,100\}\), and seeds \(\{11,23,47\}\). Three pre-registered studies (amendments 0001–0004) produced frozen-gate verdicts of "MML ≥ LMOP" under every fine-tuning protocol, and one apparent LMOP win under a zero-shot protocol. A post-hoc audit shows neither outcome is informative. The zero-shot "win" came from a degenerate across-seed confidence interval (a single checkpoint per method), and the zero-shot protocol was adopted after its outcome had been observed. More fundamentally, the shared FNO backbone has no coordinate channels, so it cannot represent the Dirichlet boundary layer. Every trained model (relative \(L_2\) 0.50–0.54) is worse than an input-blind train-mean-field predictor (0.33 ID / 0.24 OOD). On validation data, adding coordinate channels reduces Scratch error at \(N=100\) from 0.51 to 0.08. The research question therefore remains untested. We report the pipeline, the failure analysis, and a proposed repaired protocol (amendment 0005).

## 1. Question

Latent predictive pretraining on manufactured Darcy triplets may transfer to genuine numerical solutions better than direct manufactured-solution regression under small genuine-label budgets. This is a comparative transfer hypothesis, not an architecture-novelty claim (see `LITERATURE_COLLISION_MAP.md`).

## 2. Problem formulation

\[
-\nabla\cdot\bigl(a(x)\nabla u(x)\bigr)=f(x)\ \text{on }[0,1]^2,\qquad u|_{\partial\Omega}=0,
\]
discretized with cell-centered finite differences and harmonic-averaged face permeabilities on \(n=64\) interior points per axis (`darcy_reference.py`). The same operator \(A_a\) is used for manufactured pairs \(\tilde f=A_a\tilde u\) and for genuine solves with \(f\equiv 1\).

**Genuine data.** Log-GRF permeability (ID length scale 0.20, variance 1, clipped to [0.1, 10]); train pool 250, validation 50, test-ID 50. Primary OOD: correlation-length shift to 0.08 (50 examples). Splits are hashed manifests; a SHA-1 check of every permeability field finds zero overlap between train, validation, test-ID, OOD, and manufactured sets, and \(N=25\subset N=100\).

**Manufactured data (Study-2/3).** 2000 sine-mode MMS pairs, \(k_{\max}\le 4\), rescaled so \(\|u\|_2\in[1.8,3.4]\) (manufactured/genuine norm ratio 0.978).

## 3. Methods

All methods share one `FNO2d` (width 32, 12 modes, 4 layers; ≈1.19M parameters) with per-instance input normalization \(a/\mathrm{mean}(a)\), \(f/\mathrm{rms}(f)\).

- **Scratch** — genuine fine-tuning from random initialization.
- **MML-direct** — 4000 steps of relative-MSE regression \((a,\tilde f)\to\tilde u\) on the manufactured corpus.
- **LMOP-JEPA** — 4000 steps; online encoder identical to the downstream backbone; multi-block masked context \((a,f)\); EMA target encoder on \((u_n,u_n)\); 3-layer predictor; VICReg variance/covariance terms; hybrid \(u\)-head. Study-2: \(\lambda_{\mathrm{var}}=25,\lambda_u=0.1\). Study-3: \(\lambda_{\mathrm{var}}=1,\lambda_u=1\).
- **Shuffled-physics control** — LMOP trained with \(u\) permuted relative to \((a,f)\).

**Adaptation protocols.** `full_ft` (Study-2: frozen backbone for 10 epochs, then all layers, 60 epochs), `probe` (head only), `low_lr_ft` (backbone at 0.05× lr), `zero_shot` (pretrained head, no genuine labels). Model selection uses the validation split only.

**Claim gate (frozen).** Support iff LMOP beats MML in every seed with non-overlapping across-seed 95% intervals in at least one (N, distribution) cell, and shuffled pretraining loss exceeds correct.

## 4. Results

All numbers are relative \(L_2\) on genuine test sets, mean ± sample std over three seeds, reproduced from `results/tables/`. Lower is better.

### 4.1 Frozen-gate verdicts

| Study | Protocol | Evidence class | Verdict | Source |
|---|---|---|---|---|
| 1 (0001–0002) | full FT | archived, known-broken pipeline | FALSIFIES | `confirmatory.csv` |
| 2 (0003) | full FT (freeze→unfreeze) | preregistered | FALSIFIES | `claim_gate_v2.md` |
| 3 (0004) | zero_shot | **post-hoc / exploratory** | NOT_TESTABLE_NO_SEED_VARIANCE | `claim_gate_s3_zero_shot.md` |
| 3 (0004) | probe | preregistered (primary) | FALSIFIES | `claim_gate_s3_probe.md` |
| 3 (0004) | low_lr_ft | preregistered (secondary) | FALSIFIES | `claim_gate_s3_low_lr_ft.md` |

Shuffled control passed in Studies 2 and 3 (shuffled loss 15.83 > 14.58; 2.23 > 1.40).

### 4.2 Main comparison (N = 100)

| Protocol | Dist | Scratch | MML-direct | LMOP-JEPA |
|---|---|---|---|---|
| full FT (S2) | ID | 0.503±0.006 | 0.500±0.002 | 0.518±0.002 |
| probe (S3) | ID | 0.525±0.002 | 0.499±0.001 | 0.503±0.001 |
| low_lr_ft (S3) | ID | 0.500±0.002 | 0.497±0.002 | 0.502±0.002 |
| full FT (S2) | OOD | 0.538±0.002 | 0.533±0.002 | 0.544±0.007 |
| probe (S3) | OOD | 0.539±0.002 | 0.533±0.001 | 0.531±0.001 |
| low_lr_ft (S3) | OOD | 0.540±0.006 | 0.532±0.001 | 0.533±0.000 |

N = 25 tables and zero-shot are in `STUDY3_RESULTS.md`; Figure 1: `figures/confirmatory_study3_adaptations.png`.

### 4.3 Failure analysis

**Degenerate zero-shot gate.** Without fine-tuning, the MML and LMOP models are the same for every seed and N (0.7778 and 0.6857 ID in all six rows), so the across-seed interval has zero width. The original gate counted this as four "non-overlapping" wins. The corrected gate returns `NOT_TESTABLE_NO_SEED_VARIANCE`. Amendment 0004 also quotes these exact numbers in its problem statement, so zero-shot was chosen after its outcome was known. Separately, zero-shot LMOP (0.686) is worse than Scratch fine-tuned on 25 labels under every protocol (0.515–0.538).

**Trivial-baseline failure (decisive).** `FNO2d` has no coordinate channels and no padding, so it is exactly equivariant to periodic translations. With \(f\equiv1\), the model cannot locate the Dirichlet boundary.

| Test set | Train-mean field | Best spatially constant output | Trained models (N=100, seed 11) |
|---|---|---|---|
| ID | 0.335 | 0.549 | 0.500–0.522 |
| OOD | 0.240 | 0.533 | 0.532–0.538 |

Predicted fields are as large on the boundary as at the center (ratio 0.96–1.00; truth 0.06). An engineering check on the **validation** split with Scratch, \(N=100\), and the frozen optimizer gives 0.509 without coordinates and **0.084** with \((x,y)\) channels (train-mean field 0.331). Figure 2: `figures/audit_trivial_baselines.png`. Sources: `audit_positional_floor.json`, `audit_coord_sanity.json`.

**Ruled out.** Scale non-identifiability from the input normalization bounds error at ≈0.03 (`audit_scale_identifiability.json`). Split leakage: none found.

## 5. Discussion

The frozen gates executed correctly on the data they received, but in a regime where no downstream model beats an input-blind baseline. The MML-vs-LMOP gaps (≤0.03) describe small differences between models that all fail the task, and they do not measure transfer. We therefore do not claim that MML transfers better than LMOP, or the reverse. Two methodological lessons generalize. First, surrogate comparisons need a trivial-baseline gate (mean field, best constant) before any confirmatory analysis. Second, across-seed intervals are meaningless when seeds do not change the evaluated model.

## 6. Limitations

Laptop/MPS compute; \(N\in\{25,100\}\) only; one PDE family; the hybrid \(u\)-head couples JEPA to regression; four amendments in two days, each following an unfavorable result (garden-of-forking-paths risk, disclosed in each amendment); the coordinate-channel check is a single-seed validation experiment and is not claim evidence.

## 7. Proposed repair (not run)

Amendment 0005 (PROPOSED): add coordinate channels to every `FNO2d`, add a mandatory validation sanity gate (Scratch must beat the train-mean field), and keep everything else frozen. Status: **NOT RUN**, pending author approval.

## 8. Disallowed claims

No novelty claims for manufactured solutions, operator learning, or JEPA. Krylov-JEPA is unrelated. Smoke metrics are not evidence. No LMOP advantage may be claimed from zero-shot numbers.

## Reproducibility

- Study-2: `git checkout 1f0430d && ./research/free-physics-jepa/scripts/overnight_confirmatory_v2.sh`. HEAD `protocol.yaml` carries Study-3 loss weights, so Study-2 needs its commit.
- Study-3: `scripts/eval_study3.py --mode confirmatory --adaptations zero_shot --pretrain-run confirmatory_s2_v2_20260926T162730Z --skip-shuffled-retrain`, then `--adaptations probe,low_lr_ft --retrain-lmop`. Interrupted runs resume with `--resume-run <run dir>`.
- Summary and figures: `scripts/summarize_study3.py`. Audits: `scripts/audit_positional_floor.py`, `scripts/audit_coord_sanity.py`, `scripts/audit_scale_identifiability.py`.
- Environment: Python 3.14, torch 2.14 (`spectral-krylov-jepa/.venv`), Apple MPS. Run directories record the git SHA. The `low_lr_ft` tail (5 of 18 fine-tunes) ran from HEAD `22a0696`, with uncommitted changes limited to resume and gate plumbing (no training-code changes).

## References

No external references are cited yet. Related-work positioning is in `LITERATURE_COLLISION_MAP.md`, which lists topics only. Bibliographic entries must be added from primary sources before submission.
