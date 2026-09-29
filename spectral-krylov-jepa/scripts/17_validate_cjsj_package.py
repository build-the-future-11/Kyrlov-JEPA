#!/usr/bin/env python3
"""Fail-closed validator for the external CJSJ submission bundle.

This script checks only mechanical portal requirements that can be verified
without inventing signatures or reviewing scientific content.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
REQUIRED = (
    "GomezRyan_form.pdf",
    "GomezRyan_paper.docx",
    "GomezRyan_figures.ppt",
)


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"submission directory does not exist: {root}"]

    for name in REQUIRED:
        path = root / name
        if not path.is_file():
            errors.append(f"missing required file: {name}")
            continue
        size = path.stat().st_size
        if size <= 0:
            errors.append(f"empty required file: {name}")
        elif size > MAX_BYTES:
            errors.append(
                f"{name} is {size:,} bytes; CJSJ maximum is {MAX_BYTES:,} bytes"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    errors = validate(args.directory)
    if errors:
        print("CJSJ PACKAGE: BLOCKED")
        for error in errors:
            print(f"- {error}")
        print(
            "- This validator does not validate signatures, page layout, "
            "AI-disclosure completeness, or scientific claims."
        )
        return 1

    print("CJSJ PACKAGE: MECHANICAL FILE GATE PASSED")
    for name in REQUIRED:
        size = (args.directory / name).stat().st_size
        print(f"- {name}: {size:,} bytes")
    print(
        "- Human checks still required: permission signatures/fields, final "
        "paper review, AI disclosure, and portal submission."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
