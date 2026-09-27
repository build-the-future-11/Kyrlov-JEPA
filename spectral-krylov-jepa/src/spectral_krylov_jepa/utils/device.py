"""Device selection for CPU / CUDA / MPS."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Select a torch device.

    preference:
      - "auto": CUDA > MPS > CPU
      - "cpu" / "cuda" / "mps"
    """
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if pref == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if pref != "auto":
        raise ValueError(f"Unknown device preference: {preference}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS can be unstable for some ops; still preferred over CPU when available
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    return str(device)
