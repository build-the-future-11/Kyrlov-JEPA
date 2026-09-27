# Limitations

- **Synthetic potentials only** (Gaussian mixtures + specified OOD variants); not molecular Hamiltonians or experimental potentials.
- **Finite-difference discretization** on small grids (default 32×32 interior); discretization error is not extrapolated away.
- **Ground state only**; excited states and spectral densities are out of scope.
- **Pretrain/downstream mismatch:** learning short Krylov dynamics need not imply better eigenstate maps.
- **Model scale:** small transformers (≲5M params); no claim of scaling laws.
- **No quantum advantage claim**; classical ML on classical discretizations.
- **No universal PDE transfer claim**; results are for this Schrödinger family only.
- **Sign ambiguity** handled via fidelity / sign alignment; other gauges not explored.
- **Compute:** MacBook-class defaults; large runs may need cloud GPU and are separately configured.
- **Possible negative controls:** shuffled physics or remove-V may match Krylov; that would undermine the mechanistic story and must be reported.
