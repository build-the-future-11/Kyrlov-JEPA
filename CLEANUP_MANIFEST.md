# Cleanup manifest — 27 September 2026

No unique manuscript, dataset, checkpoint, experiment, historical result or user
configuration was deleted. `.serena/` was untracked at entry and remains untouched.
The original frozen protocol and decisive configuration remain unchanged.

- Restored only the two tracked test manifests overwritten by our first test run,
  from their exact HEAD bytes; all future affected tests now write manifests in
  their own temporary directories.
- Added run-local manifest output and resume identities to avoid cross-run clobbering.
- Added `.astra/` to ignore local temporary/cache/render files. Retained execution
  receipts are under `astra/2026-09-27/`.
- Removed only the temporary intermediate DOCX made by our document builder after
  incorporating its generated figure. The original official template is retained.
- Replaced template example prose in the new draft and removed invented template
  sponsor/author footnotes from that draft, preserving the source template.
- Historical outlines/status notes are retained with superseding pointers. No
  failed result or frozen outcome was erased or tuned to pass.

- Archived only this task's first figure-deck draft and generated chart workbooks
  under `.astra/`; retained the final checked figure and its validation receipt.
- Normalized generated CSV line endings to LF for Git whitespace checks; numeric
  values and ordering are unchanged.
