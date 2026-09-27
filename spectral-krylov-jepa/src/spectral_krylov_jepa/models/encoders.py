"""Shared transformer encoders for potentials and states."""

from __future__ import annotations

import torch
from torch import nn

from spectral_krylov_jepa.models.patch_embed import PatchEmbed2D, StateEmbed


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenEncoder(nn.Module):
    """Stack of transformer blocks over patch tokens + CLS."""

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_patches: int = 64,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, N, D)
        returns (cls, tokens_out)
        """
        b, n, _d = tokens.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed[:, : n + 1]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class PotentialEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed2D(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        self.encoder = TokenEncoder(
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_patches=self.patch.num_patches,
        )
        self.embed_dim = embed_dim

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch(v)
        return self.encoder(tokens)


class StateEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = StateEmbed(img_size, patch_size, embed_dim)
        self.encoder = TokenEncoder(
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_patches=self.embed.patch.num_patches,
        )
        self.embed_dim = embed_dim
        self.img_size = img_size

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.embed(q)
        return self.encoder(tokens)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def default_encoder_kwargs(size: str = "default") -> dict:
    """Architecture presets. 'smoke' is tiny; 'default' targets ~1–5M params."""
    if size == "smoke":
        return dict(embed_dim=64, depth=2, n_heads=4, mlp_ratio=2.0, patch_size=4)
    if size == "tiny":
        return dict(embed_dim=96, depth=3, n_heads=4, mlp_ratio=3.0, patch_size=4)
    if size == "default":
        return dict(embed_dim=128, depth=4, n_heads=4, mlp_ratio=4.0, patch_size=4)
    raise ValueError(f"Unknown size preset: {size}")
