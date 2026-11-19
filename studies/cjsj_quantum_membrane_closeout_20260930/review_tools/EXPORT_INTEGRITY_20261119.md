# Export integrity maintenance — 19 November 2026

## Scope set before local verification

Source: PR #12 head `5980697e7294bfc51a3a46de2daa44701220f3fd`.
The exporter and its README promise 64 comparisons and prohibit writes under
`results/`. The implementation accepts empty, partial or repeated comparison
lists, uses NumPy's default absolute tolerance despite specifying a relative
tolerance, and accepts an output directory inside the protected results tree.

This maintenance change will enforce 64 unique comparison identities with
both phases present, valid paired timing observations, a strictly relative
ratio check, and resolved output containment before creating any directory.
Synthetic fixtures cover successful export, negative cases and preservation
of input bytes. No private archive, protected outcome, trained checkpoint or
historical scientific result will be accessed or recomputed.

This is a structural/cardinality guard. The exact private archived comparison
key set is unavailable here; 64 unique identities do not independently prove
that the supplied archive is the canonical one. Existing archive hashes and
evidence-lock verification remain necessary for that separate claim.

The local scope is at most 24 synthetic test cases, two full regression test
commands and two minutes total test runtime, with one numeric thread. A failed
final check is retained and reported. No scientific campaign, paid CI,
workflow dispatch, main merge, manuscript submission or public release is
part of this work. The existing negative/inconclusive conclusions and author
review, disclosure and publication requirements remain unchanged. The commit
will retain `[skip ci]` and `skip-checks: true`.

## Verification and remaining gates

The original source failed 9 of the first 15 synthetic cases; 6 controls
passed, in 11.54 seconds. Those failures reproduced empty, partial, duplicate
and single-phase false successes, relative-tolerance mismatch, negative
timings, and direct/nested/symlink output writes inside protected results.

Before the final test command, independent static review also identified
aggregate and unit-conversion overflow, invalid interval metadata, and an
outward symlink from the sensitivity phase. Focused regressions bring the
suite to the original 24-case bound. A prospective amendment raises only the
synthetic-case cap to 28 to cover three further guard gaps: missing pairing
keys, boolean timing values, and a physical-case count inflated by an
unrelated method. The two-command/two-minute budget and all scientific,
archive-access and publication restrictions remain unchanged.

The first repaired source passed all **27 synthetic tests, zero failures or skips**
with warnings treated as errors in **9.49 seconds**. The two regression
commands used 21.03 seconds in total. Python 3.12.14, NumPy 2.3.5 and pandas
2.2.3 were used; no whole-repository or historical study test suite was run.
`git diff --check` and Python syntax compilation also passed.

The successful export control checks all 64 rows and byte-for-byte source
preservation. Negative fixtures cover incomplete/repeated/single-phase
comparisons, relative-ratio mismatch, invalid and overflowing times,
malformed interval metadata, missing paired keys, boolean timings and both
inward and outward symlink routes into consumed result directories. The
physical-case count now uses the paired methods, excluding unrelated-method
rows. A matching tiny ratio and existing-output protection are retained
controls. This verifies the exporter guard, not the archived scientific
results or their canonical identity.

Only this note, the exporter and its synthetic tests change. Frozen source,
historical receipts and scientific conclusions remain untouched. The change
is prepared on an isolated draft branch stacked onto PR #12, without main
integration, hosted CI, experiments or submission. Author review, the private
archive identity/evidence-lock checks and existing publication gates remain.

Final static review then identified an unsigned-integer subtraction gap:
accepted `uint64` timings could wrap a negative paired difference before
conversion. Before further execution, the command cap is prospectively
amended from two to three, retaining the 28-case and two-minute caps. The
minimal correction converts admitted real timing arrays to float64 before
arithmetic; one additional fixture uses exactly representable unsigned
values and a known negative difference. The preceding passing output is
retained as intermediate evidence. No scientific budget is reopened.

The final source passed **28 tests, zero failures or skips**, with warnings
treated as errors in **10.96 seconds**. All three commands used approximately
32 seconds. Independent review approved this bounded draft conditionally on
the float64 correction and that final 28-case pass; both conditions are met.
The review does not authorize parent/main merge, archive replay, CI dispatch
or research release. Remaining work is author review of the draft and the
already documented private-archive and publication gates.
