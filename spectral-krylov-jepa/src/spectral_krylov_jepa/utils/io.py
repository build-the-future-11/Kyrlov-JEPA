"""I/O and run-directory helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def project_root() -> Path:
    """Repository root (contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def git_commit_hash() -> str | None:
    try:
        root = project_root()
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def environment_info(device: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit_hash(),
        "device": device,
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except Exception:  # noqa: BLE001
        info["numpy"] = None
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except Exception:  # noqa: BLE001
        info["scipy"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:  # noqa: BLE001
        info["torch"] = None
    return info


def make_run_dir(
    base: str | Path,
    run_name: str,
    *,
    stamp: bool = True,
) -> Path:
    """Create experiments/raw/<run_name>[_timestamp]/"""
    base = Path(base)
    if stamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_name}_{ts}"
    else:
        run_id = run_name
    path = ensure_dir(base / run_id)
    return path
