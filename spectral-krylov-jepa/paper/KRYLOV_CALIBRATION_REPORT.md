# Krylov-JEPA Physics Calibration Report

**Evidence commit:** `ac06d180ac44d82e632264e9838f71cb86077482`  
**Branch:** `physics-calibration`  
**Pull request:** #7  
**Confirmatory workflow:** GitHub Actions run `36553428756`  
**Confirmatory artifact:** `krylov-confirmatory-calibration` (artifact `11024904496`)  
**Calibration smoke workflow:** run `36552976072`

## 1. Executive result

The calibration program fixes the dominant *physical residual* failure of the historical neural decoder, but it does **not** establish a useful Krylov-pretraining advantage.

The original retained 20-label result was approximately:

| Method | Fidelity | Residual at true E |
|---|---:|---:|
| Scratch | 0.993945 | 29.043635 |
| Krylov-JEPA | 0.991650 | 40.190890 |

A matched three-seed development study with the physics-aware objective and low-frequency decoder reduced the Krylov residual from a mean of about **31.44** to **3.23**. This is a real improvement in physical consistency. However, matched physics-aware scratch reached **2.72**, so the improvement comes primarily from the downstream physics calibration, not from Krylov pretraining.

A second label-free pretraining variant added a projected low-energy Ritz target directly to the transferable potential encoder. The target was numerically validated against the free-box eigensolver, but it did not improve aggregate transfer in the development study.

The fresh confirmatory study then used completely new unlabeled/labeled seed ranges, a fresh held-out ID test set, three training seeds, four label budgets, and three OOD families. Under the predeclared joint threshold

- fidelity >= 0.99,
- residual at true energy <= 5,
- relative energy error <= 0.06,

matched scratch first passes at **20 labels**, while both Krylov variants first pass at **40 labels**.

**Therefore the current evidence does not support a Krylov label-efficiency claim.**

## 2. What changed

The calibration branch adds:

1. a differentiable finite-difference Hamiltonian action matching the repository sparse Hamiltonian,
2. a Hamiltonian-residual training loss,
3. Rayleigh-energy consistency,
4. a low-frequency Dirichlet sine-basis decoder,
5. configurable physics-aware fine-tuning and evaluation,
6. a label-free projected-Ritz auxiliary target for the potential encoder,
7. numerical tests for Hamiltonian equivalence, eigenpair residuals, gradients, sine-decoder normalization, and projected-Ritz targets,
8. a matched multi-seed calibration smoke workflow,
9. a fresh confirmatory label-efficiency/OOD workflow,
10. representation-rank diagnostics and fresh classical baselines.

All new physics/calibration tests and the existing engineering suite pass on the evidence commit.

## 3. Root-cause diagnosis

The historical failure was not caused by a single bug.

### Downstream objective mismatch

The historical downstream objective rewarded fidelity, wavefunction MSE, and energy accuracy but did not directly optimize the Schrödinger residual. Small high-frequency components can preserve high overlap while being strongly amplified by the Laplacian.

Adding residual/Rayleigh terms and constraining the decoder to a low-frequency sine basis reduces this failure by roughly an order of magnitude.

### Pretraining still does not transfer an advantage

Once scratch and Krylov use the same calibrated downstream model, scratch remains at least as label-efficient. The evidence therefore points to the pretraining/downstream relationship, not simply the decoder.

### The benchmark remains highly favorable to low-complexity physics

On the fresh ID test set:

| Baseline | Labels | Fidelity | Residual at true E | Relative energy error |
|---|---:|---:|---:|---:|
| Free-box | 0 | 0.990639 | 1.726936 | 0.022879 |
| 9-mode Ritz | 0 | **0.999895** | 0.854652 | 0.001281 |
| Linear ridge | 5 | 0.997910 | 1.328752 | 0.069397 |
| Linear ridge | 10 | 0.999240 | 0.963841 | 0.026891 |
| Linear ridge | 20 | 0.999850 | **0.517054** | 0.012734 |
| Linear ridge | 40 | 0.999881 | **0.340564** | 0.020349 |

This distribution still does not require a high-capacity neural surrogate to achieve excellent ground-state accuracy.

## 4. Fresh confirmatory protocol

The confirmatory study froze the calibrated configuration before generating the fresh test corpus.

- Grid: 16 x 16 interior
- Unlabeled corpus: 240 potentials
- Train pool: 60
- Validation: 20
- Fresh ID test: 24
- OOD: 16 each for narrow, strong, and separated-double-well families
- Label budgets: 5, 10, 20, 40
- Independent pretraining/fine-tuning seeds: 11, 23, 47
- Fresh unlabeled base seed: 100000
- Fresh ID base seed: 120000
- Fresh OOD base seed: 140000
- Split seed: 92929

Methods:

- `scratch_physics`
- `krylov_physics`
- `krylov_projected_physics`
- free-box
- 9-mode Rayleigh-Ritz
- linear ridge

## 5. Fresh ID label-efficiency results

Means are over three training seeds for neural methods.

| Method | Labels | Fidelity | Residual at true E | Relative energy error |
|---|---:|---:|---:|---:|
| Scratch + physics | 5 | 0.977688 | **8.324642** | **0.076371** |
| Krylov + physics | 5 | **0.979306** | 9.075533 | 0.076840 |
| Projected Krylov + physics | 5 | 0.976223 | 10.247466 | 0.077089 |
| Scratch + physics | 10 | 0.985009 | **5.460844** | 0.076716 |
| Krylov + physics | 10 | 0.985764 | 6.363505 | 0.076960 |
| Projected Krylov + physics | 10 | **0.986864** | 6.404783 | **0.070708** |
| Scratch + physics | 20 | 0.990119 | 2.754477 | **0.057642** |
| Krylov + physics | 20 | **0.990569** | **2.619812** | 0.065228 |
| Projected Krylov + physics | 20 | 0.990420 | 2.676180 | 0.062981 |
| Scratch + physics | 40 | 0.991261 | **1.933551** | **0.048598** |
| Krylov + physics | 40 | **0.991433** | 2.010640 | 0.052671 |
| Projected Krylov + physics | 40 | 0.991354 | 1.991068 | 0.052531 |

At 20 labels, ordinary Krylov initialization has a small mean fidelity/residual advantage over scratch, but its relative energy error is worse and the advantage is not consistently dominant across metrics. In paired seed-level comparisons, the 20-label residual difference (Krylov minus scratch) is approximately -0.135 on average, while the relative-energy-error difference is approximately +0.0076. This is not sufficient for a transfer-win claim.

### Predeclared threshold result

| Method | First label budget meeting all three thresholds |
|---|---:|
| Scratch + physics | **20** |
| Krylov + physics | 40 |
| Projected Krylov + physics | 40 |

The confirmatory study therefore rejects the proposed label-efficiency win under this protocol.

## 6. OOD results at 40 labels

| Method | OOD split | Fidelity | Residual at true E | Relative energy error |
|---|---|---:|---:|---:|
| Scratch + physics | Narrow | **0.999386** | **1.837841** | 0.017979 |
| Krylov + physics | Narrow | 0.999084 | 2.351880 | **0.012060** |
| Projected Krylov + physics | Narrow | 0.998689 | 2.891220 | 0.012408 |
| Scratch + physics | Double | 0.997440 | **1.563776** | **0.027820** |
| Krylov + physics | Double | 0.997421 | 1.649275 | 0.034926 |
| Projected Krylov + physics | Double | **0.997455** | 1.636381 | 0.030904 |
| Scratch + physics | Strong | **0.943756** | **5.043699** | **1.264383** |
| Krylov + physics | Strong | 0.943497 | 5.113767 | 1.277263 |
| Projected Krylov + physics | Strong | 0.943055 | 5.044003 | 1.290742 |

There is no robust OOD Krylov advantage. The strong-amplitude family remains difficult for all neural variants, especially in energy prediction.

## 7. Representation diagnostics

Effective rank was measured on potential-encoder features before downstream fine-tuning.

Across three seeds:

| Encoder | Mean effective rank |
|---|---:|
| Random/scratch initialization | 1.901 |
| Original Krylov pretraining | 2.322 |
| Projected-Ritz Krylov pretraining | **3.406** |

Projected-Ritz pretraining also lowers off-diagonal feature covariance. Thus the negative transfer result cannot be reduced to the statement that the projected representation simply collapsed. It changes the representation and raises this rank diagnostic, but the change does not translate into better label efficiency on this task.

These diagnostics are descriptive, not a proof of semantic representation quality.

## 8. Mechanistic ablation from the bounded development study

At seed 11, the calibrated variants showed the following pattern:

- historical-style Krylov pixel decoder: residual approximately 40,
- sine decoder without physics residual: residual approximately 5.07,
- pixel decoder with residual/Rayleigh training: residual approximately 4.22,
- sine decoder with residual/Rayleigh training: residual approximately 3.35.

This supports a combined mechanism: spectral restriction removes much of the high-frequency failure, and the explicit physical objective further reduces residual. The effect is downstream-calibration driven and does not by itself establish a pretraining benefit.

## 9. Success-level audit

Using the hierarchy defined before implementation:

- **Level 1 — reduce residual versus original Krylov:** PASS.
- **Level 2 — beat matched scratch across multiple seeds on physical metrics:** NOT ESTABLISHED.
- **Level 3 — better label efficiency than scratch:** FAIL under the frozen threshold.
- **Level 4 — advantage survives OOD:** NOT ESTABLISHED.
- **Level 5 — competitive with/exceeds ridge or Ritz in a nontrivial regime:** FAIL on current benchmark.
- **Level 6 — advantage survives compute/statistical controls:** NOT REACHED.

## 10. What the data show

1. The historical residual catastrophe is largely fixable with a physics-aware objective and spectral decoder.
2. The current Krylov pretraining does not provide a reproducible label-efficiency advantage over matched scratch.
3. A direct projected-Ritz auxiliary target changes representation geometry but does not rescue transfer.
4. Simple classical/statistical controls still solve the current synthetic distribution substantially better.
5. The strong-amplitude OOD family remains a genuine failure mode for the neural energy head.

## 11. What remains a hypothesis

The following are **not** established by the current data:

- that Krylov pretraining is intrinsically useless,
- that a different low-energy filter or transfer interface cannot help,
- that the result generalizes to higher-resolution or realistic Hamiltonians,
- that the method cannot become useful once fixed low-dimensional sine/Ritz structure stops being sufficient.

## 12. Recommended next scientific experiment

Do **not** keep hyperparameter-searching this same easy ID distribution for a positive result.

The smallest useful next experiment is a preregistered harder family where:

1. the free-box control clearly degrades,
2. a fixed 9-mode Ritz subspace is no longer nearly exact,
3. the solution manifold varies substantially with the potential,
4. train/validation/test generators and thresholds are frozen before training,
5. scratch and Krylov use identical downstream architecture and compute accounting,
6. the fresh test set is untouched until the configuration is frozen.

Only then is it meaningful to ask whether unlabeled operator trajectories improve label efficiency.

## 13. Reproduction

Bounded calibration/ablation study:

```bash
python spectral-krylov-jepa/scripts/18_run_physics_calibration_smoke.py \
  --output-dir calibration_artifacts \
  --seeds 11,23,47
```

Fresh confirmatory label-efficiency/OOD study:

```bash
python spectral-krylov-jepa/scripts/19_run_confirmatory_calibration.py \
  --output-dir confirmatory_artifacts
```

Run the package tests:

```bash
python -m pytest spectral-krylov-jepa/tests tests -q
```

## 14. Claim boundary

This report supports a bounded negative/diagnostic conclusion on a finite synthetic 16 x 16 quantum ground-state benchmark. It does not establish universal superiority or inferiority of Krylov pretraining, validity on molecular electronic-structure Hamiltonians, state-of-the-art performance, or compute-matched superiority to classical eigensolvers.

The scientifically defensible result is:

> Physics-aware calibration substantially repairs the original Krylov-JEPA physical-residual failure, but under a fresh multi-seed label-efficiency test, the current Krylov pretraining does not outperform matched scratch and remains far behind simple Ritz/ridge controls on this benchmark.
