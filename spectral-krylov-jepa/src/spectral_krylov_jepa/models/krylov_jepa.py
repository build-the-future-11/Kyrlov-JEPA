"""Spectral Krylov-JEPA: predict next Lanczos state latent from (V, q0..q_{k-1})."""

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
from spectral_krylov_jepa.models.predictors import CoefficientHead, LatentPredictor


@dataclass
class KrylovJEPAOutput:
    loss: torch.Tensor
    loss_jepa: torch.Tensor
    loss_coeff: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor


class KrylovJEPA(nn.Module):
    """(V, q0, ..., q_{k-1}) → z(q_k) with optional α/β auxiliary loss.

    ``context_steps`` is the number of input Lanczos states (K in ablations):
      K1: (V, q0) → z(q1)
      K2: (V, q0, q1) → z(q2)
      K3: (V, q0, q1, q2) → z(q3)
    """

    def __init__(
        self,
        img_size: int = 32,
        size: str = "default",
        context_steps: int = 2,
        predictor_depth: int = 2,
        fuse_depth: int = 2,
        ema_momentum: float = 0.996,
        lambda_coeff: float = 0.0,
        remove_v: bool = False,
        **encoder_overrides,
    ) -> None:
        super().__init__()
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        kwargs = default_encoder_kwargs(size)
        kwargs.update(encoder_overrides)
        dim = kwargs["embed_dim"]
        self.context_steps = context_steps
        self.ema_momentum = ema_momentum
        self.lambda_coeff = lambda_coeff
        self.remove_v = remove_v
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
        # type: 0 = V, 1..context_steps = qi
        self.type_embed = nn.Parameter(torch.zeros(1 + context_steps, 1, dim))
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.step_embed = nn.Parameter(torch.zeros(context_steps, 1, dim))
        nn.init.trunc_normal_(self.step_embed, std=0.02)

        self.predictor = LatentPredictor(
            embed_dim=dim,
            depth=predictor_depth,
            n_heads=kwargs.get("n_heads", 4),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        )
        # Predict α_{k-1} and β_k for the step that produces q_k
        self.coeff_head = CoefficientHead(dim, n_coeff=2)
        self.embed_dim = dim

    def fuse_context(self, v: torch.Tensor, qs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(qs) != self.context_steps:
            raise ValueError(f"Expected {self.context_steps} context states, got {len(qs)}")
        token_list: list[torch.Tensor] = []
        cls_list: list[torch.Tensor] = []
        if not self.remove_v:
            v_cls, v_tok = self.potential_enc(v)
            v_tok = v_tok + self.type_embed[0]
            token_list.append(v_tok)
            cls_list.append(v_cls)
        for i, q in enumerate(qs):
            q_cls, q_tok = self.state_enc(q)
            q_tok = q_tok + self.type_embed[1 + i] + self.step_embed[i]
            token_list.append(q_tok)
            cls_list.append(q_cls)
        cls = torch.stack(cls_list, dim=0).mean(dim=0)
        tokens = torch.cat(token_list, dim=1)
        x = torch.cat([cls.unsqueeze(1), tokens], dim=1)
        for blk in self.fuse:
            x = blk(x)
        x = self.fuse_norm(x)
        return x[:, 0], x[:, 1:]

    def forward(
        self,
        v: torch.Tensor,
        q_context: torch.Tensor,
        q_target: torch.Tensor,
        alpha: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> KrylovJEPAOutput:
        """
        q_context: (B, context_steps, n_dof)
        q_target: (B, n_dof)  — q_{context_steps}
        alpha, beta: full depth vectors; uses index context_steps-1
        """
        qs = [q_context[:, i] for i in range(self.context_steps)]
        cls, tokens = self.fuse_context(v, qs)
        z_pred = self.predictor(cls, tokens)
        with torch.no_grad():
            z_tgt, _ = self.target_state(q_target)
            z_tgt = nn.functional.layer_norm(z_tgt, (z_tgt.shape[-1],))
        z_pred_n = nn.functional.layer_norm(z_pred, (z_pred.shape[-1],))
        loss_jepa = torch.mean((z_pred_n - z_tgt.detach()) ** 2)

        loss_coeff = torch.zeros((), device=v.device, dtype=v.dtype)
        if self.lambda_coeff > 0 and alpha is not None and beta is not None:
            # Coefficients for step j = context_steps - 1
            j = self.context_steps - 1
            coeff_true = torch.stack([alpha[:, j], beta[:, j]], dim=-1)
            coeff_hat = self.coeff_head(cls)
            loss_coeff = torch.mean((coeff_hat - coeff_true) ** 2)

        loss = loss_jepa + self.lambda_coeff * loss_coeff
        return KrylovJEPAOutput(
            loss=loss,
            loss_jepa=loss_jepa,
            loss_coeff=loss_coeff,
            z_pred=z_pred_n,
            z_target=z_tgt,
        )

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.state_enc, self.target_state, self.ema_momentum)

    def potential_encoder(self) -> PotentialEncoder:
        return self.potential_enc
