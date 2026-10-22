# Retained-evidence presentation draft — 7 October 2026

[One-page poster](poster.pdf) and [250-word abstract](abstract.txt), prepared as a nonarchival presentation candidate. Author review and venue-specific format/overlap checks remain open. This package was not submitted, and does not mark the project complete.

The renderer verifies every pinned source hash, recomputes the displayed retained arithmetic, and emits [claim data](claim_data.json) and [checks](checks.json). No model training, checkpoint replay, new outcome sweep or protected evaluation is performed. The complete final poster page was rendered and visually inspected after embedding DejaVu fonts; the initial font fallback defect was corrected.

## Rebuild

Run from this package directory with Python 3.11+, ReportLab and pypdf installed:

```sh
python -B build_materials.py --source-root ../.. --output .
```

The default font directory is `/usr/share/fonts/truetype/dejavu`. Set `CONFERENCE_FONT_DIR` to a directory containing `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` on other systems. The supplied PDF embeds its fonts and does not require them to view. Building regenerates presentation outputs only; pinned scientific sources are read-only.

## Evidence and boundary

- [experiments/summaries/smoke_latest.json](https://github.com/build-the-future-11/Kyrlov-JEPA/blob/24e871d27215020618b0febc857f6fc99dfa4e30/spectral-krylov-jepa/experiments/summaries/smoke_latest.json)
- [results/astra_baselines_20260927/baseline_audit.json](https://github.com/build-the-future-11/Kyrlov-JEPA/blob/24e871d27215020618b0febc857f6fc99dfa4e30/spectral-krylov-jepa/results/astra_baselines_20260927/baseline_audit.json)
- [paper/submission/CJSJ_DRAFT.md](https://github.com/build-the-future-11/Kyrlov-JEPA/blob/24e871d27215020618b0febc857f6fc99dfa4e30/spectral-krylov-jepa/paper/submission/CJSJ_DRAFT.md)

Input commit: `24e871d27215020618b0febc857f6fc99dfa4e30`. [Source manifest](source_manifest.json) records SHA-256 identities. The poster's empirical statements remain attributed to those retained sources; report arithmetic is not original model replay.

The retained development result does not establish useful transfer or label efficiency. Preserve the existing journal application and assess nonarchival overlap before submission.

The unbranded poster is a draft, not a claim of workshop selection or acceptance. The final venue call, authorship/presenter details and any archival restrictions must be reconciled before submission. Ordinary drafting and review work can continue within the already authorized scope.
