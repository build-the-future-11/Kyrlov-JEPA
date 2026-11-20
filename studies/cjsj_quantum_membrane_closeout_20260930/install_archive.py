#!/usr/bin/env python3
"""Install the pinned research archive in a new directory and apply build guards.

No downloads, paid runners, training, or upstream writes are performed.
The original archive is supplied with the September 30 CJSJ conversation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

EXPECTED_SHA256 = "b0f84e91e20998b0bf5b16c029504daee9fcd28095052c8219e6464d9a909972"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def install(archive: Path, destination: Path) -> Path:
    archive, destination = archive.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if sha(archive) != EXPECTED_SHA256:
        raise ValueError("Archive does not match the reviewed snapshot; no files installed")
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            p = Path(item.filename)
            if p.is_absolute() or ".." in p.parts or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Unsafe archive entry: {p}")
        destination.mkdir(parents=True)
        z.extractall(destination)
    root = destination / "cjsj_pivot"
    here = Path(__file__).resolve().parent
    for relative in ("source/manuscript_evidence.py", "verification/test_evidence_guard.py", "verification/RECEIPT.json"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(here / relative, target)
    shutil.copy2(root / "SHA256SUMS", root / "verification/INPUT_SHA256SUMS")
    paths = sorted([p for p in (root / "results").rglob("*") if p.is_file()]
                   + [root / "protocol.json", root / "figures/figure_data.json"]
                   + list((root / "figures").glob("*.png")))
    lock = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Manuscript identity lock from the pinned reviewed archive, not a correctness proof.",
            "files": {str(p.relative_to(root)): sha(p) for p in paths}}
    (root / "verification/EVIDENCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    for name in ("build_documents.py", "finalize_text.py"):
        p = root / "source" / name
        text = p.read_text()
        needle = "from pathlib import Path\n"
        if text.count(needle) != 1:
            raise RuntimeError(f"Unexpected builder structure: {name}")
        text = text.replace(needle, needle + "from manuscript_evidence import verify_evidence\n"
                            + "verify_evidence(Path(__file__).resolve().parents[1])\n", 1)
        p.write_text(text)
    print(f"Installed {root}; {len(paths)} locked inputs. Scientific source and results unchanged.")
    return root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path, help="A NEW directory; existing paths are refused")
    a = parser.parse_args()
    install(a.archive, a.destination)
