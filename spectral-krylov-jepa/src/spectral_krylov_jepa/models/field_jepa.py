"""Field-JEPA: mask patches of V and predict target latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from spectral_krylov_jepa.models.encoders import PotentialEncoder, default_encoder_kwargs
from spectral_krylov_jepa.models.ema_utils import clone_as_ema, update_ema
from spectral_krylov_jepa.models.predictors import LatentPredictor


def random_patch_mask(
    batch: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return boolean mask (B, N) where True = masked (target / hidden)."""
    n_mask = max(1, int(round(num_patches * mask_ratio)))
    noise = torch.rand(batch, num_patches, device=device, generator=generator)
    ids = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, ids[:, :n_mask], True)
    return mask


@dataclass
class FieldJEPAOutput:
    loss: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor


class FieldJEPA(nn.Module):
    """Context encoder on visible V patches; predict EMA target CLS of full V."""

    def __init__(
        self,
        img_size: int = 32,
        size: str = "default",
        mask_ratio: float = 0.4,
        predictor_depth: int = 2,
        ema_momentum: float = 0.996,
        **encoder_overrides,
    ) -> None:
        super().__init__()
        kwargs = default_encoder_kwargs(size)
        kwargs.update(encoder_overrides)
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.online = PotentialEncoder(img_size=img_size, **kwargs)
        self.target = clone_as_ema(self.online)
        self.predictor = LatentPredictor(
            embed_dim=self.online.embed_dim,
            depth=predictor_depth,
            n_heads=kwargs.get("n_heads", 4),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.online.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.num_patches = self.online.patch.num_patches

    def encode_context(self, v: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.online.patch(v)  # (B, N, D)
        # Replace masked patches with mask token for context encoder
        mask_tok = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        context_tokens = torch.where(mask.unsqueeze(-1), mask_tok, tokens)
        return self.online.encoder(context_tokens)

    def forward(self, v: torch.Tensor, mask: torch.Tensor | None = None) -> FieldJEPAOutput:
        b = v.shape[0]
        if mask is None:
            mask = random_patch_mask(b, self.num_patches, self.mask_ratio, v.device)
        cls_ctx, tok_ctx = self.encode_context(v, mask)
        z_pred = self.predictor(cls_ctx, tok_ctx)
        with torch.no_grad():
            z_tgt, _ = self.target(v)
            z_tgt = nn.functional.layer_norm(z_tgt, (z_tgt.shape[-1],))
        z_pred_n = nn.functional.layer_norm(z_pred, (z_pred.shape[-1],))
        loss = torch.mean((z_pred_n - z_tgt.detach()) ** 2)
        return FieldJEPAOutput(loss=loss, z_pred=z_pred_n, z_target=z_tgt)

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.online, self.target, self.ema_momentum)

    def potential_encoder(self) -> PotentialEncoder:
        return self.online
