"""Patch embedding for 2D fields and wavefunction states."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """Non-overlapping 2D patch embedding via strided convolution."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) or (B, H, W)

        Returns
        -------
        tokens : (B, N, D)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            raise ValueError(
                f"Expected spatial size {(self.img_size, self.img_size)}, got {tuple(x.shape[-2:])}"
            )
        x = self.proj(x)  # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return self.norm(x)


class StateEmbed(nn.Module):
    """Embed a flattened state vector as a spatial field then patch-embed."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.n_dof = img_size * img_size
        self.patch = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q : (B, n_dof) or (B, H, W)
        returns (B, N, D)
        """
        if q.ndim == 2:
            q = q.view(q.shape[0], self.img_size, self.img_size)
        return self.patch(q)
