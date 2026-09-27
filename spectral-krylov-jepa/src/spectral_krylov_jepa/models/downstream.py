"""Downstream ground-state predictor: V → (E0, ψ0)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spectral_krylov_jepa.models.encoders import PotentialEncoder, default_encoder_kwargs
from spectral_krylov_jepa.models.heads import EnergyHead, WavefunctionDecoder, normalize_wavefunction_torch


@dataclass
class DownstreamOutput:
    energy: torch.Tensor
    psi: torch.Tensor  # normalized
    z: torch.Tensor
    tokens: torch.Tensor | None = None


class DownstreamGroundStateModel(nn.Module):
    """Shared downstream architecture for all methods."""

    def __init__(
        self,
        img_size: int = 32,
        size: str = "default",
        cell_area: float = 1.0 / (33 * 33),
        encoder: PotentialEncoder | None = None,
        **encoder_overrides,
    ) -> None:
        super().__init__()
        kwargs = default_encoder_kwargs(size)
        kwargs.update(encoder_overrides)
        self.encoder = encoder if encoder is not None else PotentialEncoder(img_size=img_size, **kwargs)
        dim = self.encoder.embed_dim
        patch_size = self.encoder.patch.patch_size
        self.energy_head = EnergyHead(dim)
        self.psi_decoder = WavefunctionDecoder(dim, img_size=img_size, patch_size=patch_size)
        self.cell_area = cell_area
        self.img_size = img_size
        # Optional input standardization of V (identity until set)
        self.register_buffer("v_center", torch.tensor(0.0))
        self.register_buffer("v_scale", torch.tensor(1.0))

    def set_potential_stats(self, center: float, scale: float) -> None:
        self.v_center.fill_(float(center))
        self.v_scale.fill_(max(float(scale), 1e-6))

    def set_energy_stats(self, mean: float, std: float) -> None:
        self.energy_head.set_energy_stats(mean, std)

    def _prep_v(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.v_center) / self.v_scale.clamp_min(1e-6)
    def forward(self, v: torch.Tensor) -> DownstreamOutput:
        z, tokens = self.encoder(self._prep_v(v))
        energy = self.energy_head(z)
        psi_raw = self.psi_decoder(z, tokens)
        psi = normalize_wavefunction_torch(psi_raw, cell_area=self.cell_area)
        return DownstreamOutput(energy=energy, psi=psi, z=z, tokens=tokens)

    def load_pretrained_encoder(self, encoder: PotentialEncoder, *, strict: bool = True) -> None:
        self.encoder.load_state_dict(encoder.state_dict(), strict=strict)
