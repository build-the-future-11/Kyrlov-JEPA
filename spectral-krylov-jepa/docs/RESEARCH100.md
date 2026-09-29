# Research-100 runner

This runner operationalizes all 100 ideas in the Krylov-JEPA research roadmap.

## Run everything

From the `spectral-krylov-jepa` directory:

```bash
python scripts/20_run_research100_suite.py --mode full --ids all --continue-on-error
```

The run writes a timestamped directory under `results/` containing:

- `experiments.json` and `experiments.csv`: one evidence row for each numbered idea;
- `summary.json`: execution coverage and failures;
- `RESEARCH100_REPORT.md`: human-readable registry/status report;
- difficulty, scaling, and phase-diagram CSVs when their handlers run;
- publication-style diagnostic figures under `figures/`.

## Modes

`--mode smoke` uses a tiny grid/data budget and is intended for CI and code-path verification.
`--mode full` uses larger shared datasets and longer trainable prototypes.

Each result includes a maturity tag: `analysis`, `benchmark`, `prototype`, or `trainable`.
A successful prototype means the proposed mechanism executes and emits real metrics; it does not by itself establish an efficacy claim.

## Run selected ideas

```bash
python scripts/20_run_research100_suite.py --mode full --ids 1-20,45-64,81-100 --continue-on-error
```

The full registry is embedded directly in the runner so IDs remain stable.
