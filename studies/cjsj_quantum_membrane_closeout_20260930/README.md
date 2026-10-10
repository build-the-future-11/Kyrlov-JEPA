# CJSJ quantum–membrane closeout

This directory records the separate verification pass for **Residual Corrections across Quantum and Membrane Eigenproblems**. It does not replace or reinterpret the older frozen Krylov-JEPA study on `main`.

## Scientific disposition

The archived matched-backbone JEPA/classical time ratio is 1.970758; no JEPA advantage is established. The separately tuned membrane comparison gives 0.962120 with a 95% interval [0.912966, 1.012336]; a primary speedup is not established. A narrower secondary two-term-vs-one-term correction improvement remains. Framework portability is not pretrained weight transfer.

## Verified in this pass

- All 121 original archive hashes and 35 original tests passed.
- Recalculated 64 comparisons from 36,096 stored timing observations.
- Replayed 544 large-grid solves: 528 accuracy passes and 16 retained fixed-basis failures. Used every one of 24 archived neural checkpoints.
- Six independently assembled dense small-grid checks passed.
- Replayed two training configurations. Parameters were not bitwise identical. Thirty-two additional paired functional solve checks passed; this is not a proof of model equivalence or external replication.
- Added a manuscript evidence guard and seven tests: 42 total passing tests.
- Preserved all 75 archived result files and 11 frozen scientific source files.

Detailed provenance and limitations are in `verification/RECEIPT.json`. Full raw replays, retrained checkpoints, revised manuscript, PDF, figure deck and disclosure records are distributed in the **CJSJ_Verified_Closeout_Package.zip** delivered with the conversation. They are not all stored in this lightweight PR.

## What this PR implements

The original document/report builders embed a scientific narrative with fixed numerical claims. Those claims match the reviewed archive, but silently rebuilding them after changing evidence could produce stale prose. `source/manuscript_evidence.py` now rejects changed/missing evidence before either builder writes. This is an artifact-identity safeguard, not a correctness or novelty certificate.

`install_archive.py` accepts ONLY the original reviewed `CJSJ_Quantum_Membrane_Executed_Package.zip`, checks SHA-256, safely extracts into a NEW directory, adds the guard/tests and patches the two builders. It neither downloads data nor runs training or paid jobs. Large checkpoints and raw data stay in the supplied archive instead of being duplicated in Git.

```bash
python install_archive.py /path/to/CJSJ_Quantum_Membrane_Executed_Package.zip /path/to/new-review-copy
cd /path/to/new-review-copy/cjsj_pivot
# Install source/requirements.txt in an isolated, authorized environment first.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q source/test_pivot.py source/legacy_tests verification/test_evidence_guard.py
python source/manuscript_evidence.py
python source/build_documents.py
```

The original ZIP SHA-256 is `b0f84e91e20998b0bf5b16c029504daee9fcd28095052c8219e6464d9a909972`. A different archive is deliberately refused. The installer leaves the numerical results and scientific source unchanged. The delivered verified-closeout ZIP already contains these safeguards plus the review files; it does not need this installer.

## Budget and publication boundary

This review commit carries `[skip ci]` and `skip-checks: true`. No paid GitHub Actions run is authorized. Do not remove the skip instructions or launch workflows without an approved budget. No main-branch merge, public release or journal submission is authorized by this PR.

The manuscript is a candidate requiring author review, a signed official permission form and resolution of full AI-prompt Methods disclosure. No signature, mentor consent, submission receipt or acceptance is asserted.
