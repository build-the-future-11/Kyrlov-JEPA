# Limitations

The current scientific evidence is a post-hoc audit of one development run,
not the frozen decisive experiment. Synthetic Gaussian potentials on a 16-grid
are close to a fixed free-box state; fidelity alone understates physical error.
Only one historical training seed is available. There are 12 ID test potentials
and eight per OOD family. No experimental or molecular dataset is evaluated.

Field-JEPA is an adapted full-CLS objective. It is not canonical I-JEPA.
Pretraining costs differ across Field, Operator and Krylov; equal step counts
are not equal FLOPs. Normalized-powers controls match neural shapes and parameters
but not total generation cost. Classical Ritz controls access H at inference.
Timing on a busy shared machine cannot establish a hardware-normalized speedup.
Peak memory in control summary is for the whole process, not per-model device memory.

Removing V prevents training the transferred potential encoder. A coefficient
loss on unscaled finite-difference coefficients can dominate the latent objective.
Layer normalization inside encoders remains even when the final latent loss
normalization is disabled. Low prediction loss does not rule out collapse.
Representation covariance/probe studies, independent reproduction, full multi-seed
convergence, equal-total-compute efficacy and larger-grid robustness remain open.

The control verification matrix is deliberately tiny (8-grid, 4 steps, 2 epochs,
2 examples per evaluation split). It establishes code coverage only. Full controls
and the original main study are ready to invoke but not completed. Any new
hypothesis or changed benchmark after the audit must receive a successor protocol.
The journal draft requires human authorship/eligibility/consent and final review.
