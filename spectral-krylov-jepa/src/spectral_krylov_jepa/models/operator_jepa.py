"""Operator-JEPA: predict latent of normalize(Hq) from (V, q)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spectral_krylov_jepa.models.encoders import (
    PotentialEncoder,
    StateEncoder,
    TransformerBlock,
    default_encoder_kwargs,
)
from spectral_krylov_jepa.models.ema_utils import clone_as_ema, update_ema
from spectral_krylov_jepa.models.predictors import LatentPredictor


@dataclass
class OperatorJEPAOutput:
    loss: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor


class OperatorJEPA(nn.Module):
    """(V, q) → z(normalize(H q)) via fused tokens + EMA target state encoder."""

    def __init__(
        self,
        img_size: int = 32,
        size: str = "default",
        predictor_depth: int = 2,
        fuse_depth: int = 2,
        ema_momentum: float = 0.996,
        **encoder_overrides,
    ) -> None:
        super().__init__()
        kwargs = default_encoder_kwargs(size)
        kwargs.update(encoder_overrides)
        dim = kwargs["embed_dim"]
        self.ema_momentum = ema_momentum
        self.img_size = img_size
        self.potential_enc = PotentialEncoder(img_size=img_size, **kwargs)
        self.state_enc = StateEncoder(img_size=img_size, **kwargs)
        self.target_state = clone_as_ema(self.state_enc)
        self.fuse = nn.ModuleList(
            [
                TransformerBlock(dim, kwargs.get("n_heads", 4), kwargs.get("mlp_ratio", 4.0))
                for _ in range(fuse_depth)
            ]
        )
        self.fuse_norm = nn.LayerNorm(dim)
        self.type_embed = nn.Parameter(torch.zeros(2, 1, dim))  # 0=V, 1=q
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.predictor = LatentPredictor(
            embed_dim=dim,
            depth=predictor_depth,
            n_heads=kwargs.get("n_heads", 4),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        )
        self.embed_dim = dim

    def fuse_context(self, v: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_cls, v_tok = self.potential_enc(v)
        q_cls, q_tok = self.state_enc(q)
        v_tok = v_tok + self.type_embed[0]
        q_tok = q_tok + self.type_embed[1]
        # Use mean of CLS as query seed + concatenated tokens
        cls = 0.5 * (v_cls + q_cls)
        tokens = torch.cat([v_tok, q_tok], dim=1)
        x = torch.cat([cls.unsqueeze(1), tokens], dim=1)
        for blk in self.fuse:
            x = blk(x)
        x = self.fuse_norm(x)
        return x[:, 0], x[:, 1:]

    def forward(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        hq_normalized: torch.Tensor,
    ) -> OperatorJEPAOutput:
        cls, tokens = self.fuse_context(v, q)
        z_pred = self.predictor(cls, tokens)
        with torch.no_grad():
            z_tgt, _ = self.target_state(hq_normalized)
            z_tgt = nn.functional.layer_norm(z_tgt, (z_tgt.shape[-1],))
        z_pred_n = nn.functional.layer_norm(z_pred, (z_pred.shape[-1],))
        loss = torch.mean((z_pred_n - z_tgt.detach()) ** 2)
        return OperatorJEPAOutput(loss=loss, z_pred=z_pred_n, z_target=z_tgt)

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.state_enc, self.target_state, self.ema_momentum)

    def potential_encoder(self) -> PotentialEncoder:
        return self.potential_enc
