"""Content-bound run identities; refuse mixed-source/config/data resumes."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    paths += [root / "paper/experiment_protocol.md"]
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def freeze_or_check(path: Path, payload: dict) -> None:
    canonical = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Refusing incompatible resume: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(canonical)


@contextmanager
def exclusive_run(path: Path):
    """POSIX advisory lock; OS releases it on crash, persistent file is retained."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another runner holds {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
