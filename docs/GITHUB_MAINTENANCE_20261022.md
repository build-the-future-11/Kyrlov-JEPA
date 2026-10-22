# GitHub maintenance and engineering verification — 22 October 2026

This integration keeps the LMOP frozen-matrix claim-gate repair from PR20 and makes the automatic verification boundary explicit. Existing scientific campaigns still require their own scientific authorization and are available only through an explicit manual dispatch. This maintenance action dispatches none of them.

## Engineering gates retained

The complete Krylov engineering test collection and critical lint commands remain unchanged. Its checkout is bound to the exact pull-request head (or push SHA), and a separate step verifies that identity before installation and execution. Existing hybrid mathematical tests, recycled-control invariant tests and Research-100 registry tests also remain automatic. Scientific execution, scientific result validation and scientific artifact upload are guarded by the workflow_dispatch event. The original command bodies, seeds, budgets and scientific policy are preserved.

## Claim-gate review and local evidence

PR20 source head: `1d3c56cdaca428e60d9f50e2bca9d27ba83e23f8`.

The full three-file patch was independently reviewed. Exactly one finite nonnegative primary metric is required per declared method/seed/label-budget/distribution cell. Duplicate, unexpected, missing or invalid cells cannot produce scientific comparisons. CSV ambiguity and nonfinite/duplicate-key shuffled-control JSON fail closed. The statistical formula, frozen axes, existing retained results and scientific conclusions are unchanged.

The exact source and test blobs passed 51 constructed-fixture tests in 0.12 seconds under Python3.12.14, pytest8.4.2. Critical E9/F821/F822/F823 lint also passed. This is independent software verification; retained scientific tables were not rerun, training was not performed, and protected outcomes were not opened. Existing historical verification in CLAIM_GATE_VALIDATION_2026-10-08.md is preserved with its original scope.

All eight workflow documents parse as YAML. Source review confirms the existing engineering test and lint commands are retained. New exact-head hosted engineering verification and independent review are still required before merge; no hosted success is claimed in this preparation record.

## Independent review correction

The first workflow preparation accidentally left two `if` keys on the Research-100 upload step. A permissive YAML loader accepted that mapping; independent parent review correctly rejected it. The final revision combines them into one `always() && github.event_name == 'workflow_dispatch'` expression. All eight final workflows pass a loader that rejects duplicate mapping keys. The initial validation limitation is retained here; the earlier candidate was not published.
