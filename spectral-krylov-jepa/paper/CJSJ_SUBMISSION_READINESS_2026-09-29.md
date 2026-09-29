# CJSJ submission readiness — 29 September 2026

This document separates **CJSJ packaging readiness** from the larger scientific
completion gate for Spectral Krylov-JEPA. The current CJSJ manuscript is a
bounded negative/audit report: it does not rely on the unexecuted full 192-cell
main matrix or 672-cell supplement to claim a positive Krylov advantage.

## Current official CJSJ requirements

Verified against the CJSJ 2026-2027 Guidelines and Submission Portal on
2026-09-29:

- Original research: 2–3 manuscript pages; figures and citations are excluded
  from that page limit.
- Use the official Research Paper Template.
- Upload a completed Permission to Publish form.
- Upload filenames exactly as:
  - `LastnameFirstname_form.pdf`
  - `LastnameFirstname_paper.docx`
  - `LastnameFirstname_figures.ppt`
- Maximum file size: 10 MB per file.
- AI-assisted work must be disclosed; CJSJ asks for the AI tool/version and the
  full production prompt in the detailed disclosure.
- Deadline: 2026-09-30 11:59 PM EST.

Sources:
- https://cjsjournal.squarespace.com/guidelines
- https://cjsjournal.squarespace.com/submit
- https://cjsjournal.squarespace.com/faq

## Submission-bundle verification performed 2026-09-29

A private/local submission bundle was inspected without committing it to this
public repository.

- `GomezRyan_paper.docx`: 175,726 bytes.
  - Rendered cleanly to four physical pages.
  - Pages 1–2 contain the manuscript text.
  - Page 3 contains Figure 1 and its caption.
  - Page 4 contains references only.
  - Therefore the manuscript body is two pages under the CJSJ counting rule.
- `GomezRyan_figures.ppt`: 766,464 bytes and correctly named/formatted.
- `GomezRyan_AI_prompt_disclosure.txt`: 27,514 bytes. It records the retained
  project-construction prompt, the four retained CJSJ editing prompts, and the
  available tool/version information.
- No completed `GomezRyan_form.pdf` was found in the searched private library.

All located files are below CJSJ's 10 MB per-file cap.

## Hard human gate

The remaining application artifact is a completed Permission to Publish form.
Do **not** fabricate signatures. CJSJ says original-research submitters without
a PI/mentor may leave mentor-specific fields blank, but the form itself is still
required.

The final portal submission must also be performed by the primary author. Do not
create duplicate submissions; CJSJ explicitly warns against resubmitting the
same 2026-2027 application.

## AI-disclosure gate

The manuscript already discloses AI assistance and names the four retained CJSJ
editing prompts. The companion disclosure contains the much longer retained
project-construction prompt. Before pressing Submit, confirm that the portal
provides a place for the companion disclosure or that CJSJ has otherwise
accepted that disclosure route. If not, contact CJSJ rather than silently
dropping the required prompt history.

## Scientific-claim boundary

The current CJSJ paper can be evaluated as a negative/audit result because its
claims are explicitly limited to the preserved single-seed development corpus.
The following remain necessary for a stronger positive/efficacy claim, but are
**not silently treated as completed for this CJSJ package**:

- original full experiment: 192 evaluations;
- supplementary full controls: 672 evaluations;
- independent pretraining-seed sensitivity;
- full representation-collapse/probe analysis;
- equal-total-compute efficacy;
- independent reproduction.

No submission artifact should imply that these runs have completed.

## Local final-package validator

Run:

```sh
python spectral-krylov-jepa/scripts/17_validate_cjsj_package.py /path/to/final/submission
```

The validator checks required filenames, extensions, non-empty files, and the
10 MB cap. It deliberately does **not** claim to validate signatures or
scientific content.
