"""Fail closed before rebuilding prose whose numbers were checked against an archive.

The prose is a versioned scientific narrative, not a generic results dashboard.
A result change requires reanalysis AND an explicit manuscript revision. This
identity guard prevents silently publishing old numeric text with new figures;
it does not certify methodology, source accuracy, or publication readiness.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

class EvidenceMismatch(RuntimeError):
    """A manuscript input no longer matches its reviewed evidence snapshot."""

def verify_evidence(root: Path, lock_path: Path | None = None) -> dict:
    root = Path(root).resolve()
    lock_path = Path(lock_path) if lock_path else root / "verification/EVIDENCE_LOCK.json"
    if not lock_path.is_file():
        raise EvidenceMismatch(f"Missing manuscript evidence lock: {lock_path}")
    try:
        lock = json.loads(lock_path.read_text())
        entries = lock["files"]
        if lock.get("schema") != 1 or not isinstance(entries, dict) or not entries:
            raise ValueError("Unsupported or empty lock")
    except (ValueError, KeyError, TypeError) as exc:
        raise EvidenceMismatch("Malformed manuscript evidence lock") from exc
    problems = []
    for relative, expected in entries.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise EvidenceMismatch(f"Unsafe evidence path: {relative}")
        if not target.is_file():
            problems.append(f"missing: {relative}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            problems.append(f"changed: {relative}")
    if problems:
        raise EvidenceMismatch("Refusing stale manuscript build. Reconcile data, analysis, "
                               "figures and prose before deliberately revising the lock.\n" + "\n".join(problems))
    return {"status": "PASS", "locked_inputs": len(entries), "lock": str(lock_path)}

if __name__ == "__main__":
    print(json.dumps(verify_evidence(Path(__file__).resolve().parents[1]), indent=2))
