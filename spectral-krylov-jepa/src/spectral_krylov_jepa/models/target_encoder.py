"""Target-encoder factory (EMA freeze helper re-exports)."""

from __future__ import annotations

from spectral_krylov_jepa.models.ema_utils import clone_as_ema, update_ema

__all__ = ["clone_as_ema", "update_ema"]
