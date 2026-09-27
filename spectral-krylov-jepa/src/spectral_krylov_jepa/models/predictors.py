"""Predictor networks for JEPA latent prediction."""

from __future__ import annotations

import torch
from torch import nn

from spectral_krylov_jepa.models.encoders import TransformerBlock


class LatentPredictor(nn.Module):
    """MLP / small transformer predictor from context tokens to target latent."""

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, context_cls: torch.Tensor, context_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """
        context_cls: (B, D)
        context_tokens: optional (B, N, D) — if provided, run transformer over [cls|tokens]
        returns predicted latent (B, D)
        """
        if context_tokens is None:
            x = context_cls.unsqueeze(1)
        else:
            x = torch.cat([context_cls.unsqueeze(1), context_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class CoefficientHead(nn.Module):
    """Optional auxiliary head predicting Lanczos α/β coefficients."""

    def __init__(self, embed_dim: int, n_coeff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_coeff),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
