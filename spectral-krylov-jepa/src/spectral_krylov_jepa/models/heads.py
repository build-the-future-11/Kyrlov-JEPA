"""Downstream prediction heads for energy and wavefunction."""

from __future__ import annotations

import math

import torch
from torch import nn


class EnergyHead(nn.Module):
    """MLP energy head with z-score denormalization via buffers."""

    def __init__(self, embed_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(embed_dim, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_std", torch.tensor(1.0))

    def set_energy_stats(self, mean: float, std: float) -> None:
        self.energy_mean.fill_(float(mean))
        self.energy_std.fill_(max(float(std), 1e-6))

    @property
    def use_standardization(self) -> bool:
        return not (
            float(self.energy_mean.item()) == 0.0 and abs(float(self.energy_std.item()) - 1.0) < 1e-12
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z).squeeze(-1)
        return raw * self.energy_std + self.energy_mean

    def loss_space(self, e_physical: torch.Tensor) -> torch.Tensor:
        return (e_physical - self.energy_mean) / self.energy_std.clamp_min(1e-6)


class WavefunctionDecoder(nn.Module):
    """Decode CLS and optional patch tokens to an unrestricted spatial wavefunction."""

    def __init__(
        self,
        embed_dim: int,
        img_size: int = 32,
        patch_size: int = 4,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        hidden = hidden or embed_dim * 2
        n_patches = self.grid * self.grid
        self.cls_fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_patches * patch_size * patch_size),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size * patch_size),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        b = z.shape[0]
        from_cls = self.cls_fc(z).view(
            b, self.grid, self.grid, self.patch_size, self.patch_size
        )
        from_cls = from_cls.permute(0, 1, 3, 2, 4).contiguous().view(
            b, self.img_size, self.img_size
        )

        if tokens is not None:
            n = tokens.shape[1]
            expected = self.grid * self.grid
            if n != expected:
                raise ValueError(f"Expected {expected} tokens, got {n}")
            patch = self.token_proj(tokens).view(
                b, self.grid, self.grid, self.patch_size, self.patch_size
            )
            from_tok = patch.permute(0, 1, 3, 2, 4).contiguous().view(
                b, self.img_size, self.img_size
            )
            x = from_cls + from_tok
        else:
            x = from_cls

        return self.refine(x.unsqueeze(1)).squeeze(1)


class SineBasisWavefunctionDecoder(nn.Module):
    """Decode a state in a low-frequency Dirichlet sine basis.

    n_modes is the total number of tensor-product basis functions and must be a
    perfect square: 9 means modes 1..3 in each dimension, 25 means 1..5, etc.
    """

    def __init__(
        self,
        embed_dim: int,
        img_size: int = 32,
        n_modes: int = 25,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        side = int(round(math.sqrt(n_modes)))
        if side * side != int(n_modes):
            raise ValueError(f"n_modes must be a perfect square, got {n_modes}")
        if side > img_size:
            raise ValueError(f"basis side {side} cannot exceed img_size={img_size}")

        self.img_size = int(img_size)
        self.n_modes = int(n_modes)
        self.side = side
        hidden = hidden or max(embed_dim, 64)

        coords = torch.arange(1, img_size + 1, dtype=torch.float32) / float(img_size + 1)
        basis = []
        for my in range(1, side + 1):
            sy = torch.sin(math.pi * my * coords)
            for mx in range(1, side + 1):
                sx = torch.sin(math.pi * mx * coords)
                mode = torch.outer(sy, sx)
                mode = mode / torch.linalg.vector_norm(mode).clamp_min(1e-12)
                basis.append(mode)
        self.register_buffer("basis", torch.stack(basis, dim=0))

        self.token_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_modes),
        )

    def forward(self, z: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        features = z
        if tokens is not None:
            features = features + self.token_pool(tokens.mean(dim=1))
        coeffs = self.coefficient_head(features)
        return torch.einsum("bm,mhw->bhw", coeffs, self.basis)


def normalize_wavefunction_torch(psi: torch.Tensor, cell_area: float = 1.0) -> torch.Tensor:
    """Normalize so sum(psi^2) times cell_area equals one, batch-wise."""
    flat = psi.reshape(psi.shape[0], -1)
    norms = torch.sqrt(torch.sum(flat * flat, dim=-1, keepdim=True) * cell_area + 1e-12)
    return (flat / norms).reshape_as(psi)
