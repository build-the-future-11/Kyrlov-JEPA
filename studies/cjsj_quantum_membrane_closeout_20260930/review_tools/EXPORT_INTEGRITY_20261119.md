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


## Independent premerge engineering review — 19 November 2026

The current maintenance task authorizes a bounded source review and correction
on the existing review branch. It does not adopt scientific claims, publish the
private archive, merge the parent study to main, or satisfy author and release
requirements. The historical scope and test receipts above are retained.

At source `a39b41d9c696b63fce815643ad6ee6d6a75eccdf`, all 28 existing synthetic
tests passed in 16.74 seconds with warnings as errors. A separate constructed
counterexample then showed that admitted int64 timings `2**53` and `2**53 + 1`
produced a 64-row table with a zero paired difference instead of -1000 ms.
Both observations rounded to the same float64 value before subtraction.

The correction computes each integer paired difference and its sum with Python
integers, then divides once by the paired count. This preserves signed small
differences and cancellation across pairs before scaling. The existing numeric
admission, float-observation path, ratio tolerance (`rtol=1e-12`, `atol=0`),
interval requirements, comparison identities and output protections are
unchanged. No scientific threshold or archived numerical value is changed.

Five added synthetic cases cover positive and negative adjacent int64/uint64
observations and cancellation between large opposing differences. They also
check 64-row completeness and input byte preservation. The final proposed
source passed **33 tests, zero failures or skips**, with warnings as errors,
in **15.26 seconds** on Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and pytest
9.1.1, using one numeric thread. The original 28 tests remain unchanged.

Only the exporter, its synthetic test file and this appended note change. No
private archive, frozen study source, historical receipt, protected outcome or
checkpoint was accessed or changed; no scientific run, workflow dispatch,
submission or release was performed by this review. Exact canonical archive
membership and the existing author, disclosure and publication gates remain
separate from this engineering qualification.


### Coordinator integration and continuous verification

The coordinator independently reviewed the complete exporter, all 33 synthetic
cases and the exact precision delta. A separate five-minute job is appended to
the existing engineering workflow so the nested suite is continuously checked.
It uses Python 3.12 and the tested NumPy 2.3.5, pandas 2.2.3 and pytest 9.1.1
versions, with one numeric thread and warnings as errors. Existing workflow
triggers and the original engineering job remain unchanged. The new job reads
only generated temporary fixtures, never the private archive or retained study
results. The exact published successor must pass this and all existing normal
checks before source integration into the current CJSJ review branch.

The final repair spans the same three maintenance files plus that ordinary
engineering workflow. This does not merge the parent study to main or grant
scientific, authorship, disclosure, archive-access or publication approval.

## Input snapshot correction — 20 November 2026

Base: consolidated CJSJ review head
`a89d89532267cd01e75e1418dd0f67d647553a4c`.

The exporter parsed each CSV from a path, then reopened the path to calculate
its SHA-256. It also parsed the same file again for each comparison. A path
replacement between these operations could attach the replacement's digest
to values computed from the earlier bytes and combine multiple snapshots
in one table. This is an engineering provenance defect, not a finding about
the retained scientific measurements.

The exporter now reads each timing CSV once, parses those bytes through an
in-memory stream and hashes that same byte snapshot. Every comparison that
uses the file reuses the parsed snapshot and digest. The table schema,
comparison calculations, admission checks and protected-output restrictions
remain unchanged. This does not establish canonical archive membership or
an atomic snapshot across different files.

Two synthetic regression cases replace or remove the input path immediately
after its first parse. Both fail on the original exporter: replacement
produces the wrong source digest; removal raises FileNotFoundError during
the later hash read. After the repair, all 35 synthetic cases pass with
warnings treated as errors in 9.69 seconds. The cases check all 64 output
rows and require every affected comparison to retain the original digest,
mean times and paired difference. The earlier 33 cases are retained.

Environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and pytest 9.2.0;
one numeric thread. Automatic loading of unrelated third-party pytest
plugins was disabled after the environment's AnyIO plugin emitted a pytest
deprecation during startup. No test was skipped or warning suppressed in
the executed suite. Git whitespace checks passed.

Only this note, the exporter and its synthetic tests change. No private
archive, scientific result, checkpoint or frozen study source was accessed.
The correction is a draft stacked on the existing review branch; its commit
retains `[skip ci]` and `skip-checks: true`. No paid workflow, main merge,
study run, submission or scientific release is authorized by this receipt.
