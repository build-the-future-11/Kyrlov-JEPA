"""Spectral Krylov-JEPA with optional low-energy projected-Ritz supervision."""

from __future__ import annotations

import math
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
    loss_projected_ritz: torch.Tensor
    z_pred: torch.Tensor
    z_target: torch.Tensor


class KrylovJEPA(nn.Module):
    """Predict later Lanczos-state latents from a potential and Krylov context.

    The optional projected-Ritz auxiliary task is label-free. It constructs a
    small fixed Dirichlet sine basis, projects the Hamiltonian into that basis,
    and asks the potential encoder alone to predict the lowest projected energy
    and its coefficient vector. This directly pressures the transferable
    potential encoder to preserve low-energy spectral information.
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
        lambda_projected_ritz: float = 0.0,
        projected_modes: int = 9,
        remove_v: bool = False,
        normalize_latents: bool = True,
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
        self.lambda_projected_ritz = lambda_projected_ritz
        self.remove_v = remove_v
        self.normalize_latents = normalize_latents
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
        self.coeff_head = CoefficientHead(dim, n_coeff=2)
        self.embed_dim = dim

        side = int(round(math.sqrt(projected_modes)))
        if side * side != int(projected_modes):
            raise ValueError(f"projected_modes must be a perfect square, got {projected_modes}")
        self.projected_modes = int(projected_modes)
        coords = torch.arange(1, img_size + 1, dtype=torch.float32) / float(img_size + 1)
        basis = []
        kinetic = []
        h = 1.0 / float(img_size + 1)
        for my in range(1, side + 1):
            sy = torch.sin(math.pi * my * coords)
            for mx in range(1, side + 1):
                sx = torch.sin(math.pi * mx * coords)
                mode = torch.outer(sy, sx)
                mode = mode / torch.linalg.vector_norm(mode).clamp_min(1e-12)
                basis.append(mode.reshape(-1))
                lam = (2.0 / (h * h)) * (
                    math.sin(math.pi * mx / (2.0 * (img_size + 1))) ** 2
                    + math.sin(math.pi * my / (2.0 * (img_size + 1))) ** 2
                )
                kinetic.append(lam)
        self.register_buffer("projected_basis", torch.stack(basis, dim=0))
        self.register_buffer("projected_kinetic", torch.tensor(kinetic, dtype=torch.float32))
        self.ritz_energy_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(dim, 64)),
            nn.GELU(),
            nn.Linear(max(dim, 64), 1),
        )
        self.ritz_coeff_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(dim, 64)),
            nn.GELU(),
            nn.Linear(max(dim, 64), self.projected_modes),
        )

    def fuse_context(
        self,
        v: torch.Tensor,
        qs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(qs) != self.context_steps:
            raise ValueError(f"Expected {self.context_steps} context states, got {len(qs)}")
        token_list: list[torch.Tensor] = []
        cls_list: list[torch.Tensor] = []
        v_cls: torch.Tensor | None = None
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
        return x[:, 0], x[:, 1:], v_cls

    def _projected_ritz_targets(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lowest energy and coefficient vector in the fixed sine basis."""
        b = self.projected_basis.to(dtype=v.dtype, device=v.device)
        v_flat = v.reshape(v.shape[0], -1)
        v_proj = torch.einsum("mp,bp,np->bmn", b, v_flat, b)
        h_proj = v_proj + torch.diag(self.projected_kinetic.to(dtype=v.dtype, device=v.device))
        evals, evecs = torch.linalg.eigh(h_proj)
        energy = evals[:, 0]
        coeff = evecs[:, :, 0]
        sign = torch.where(
            coeff[:, :1] < 0,
            -torch.ones_like(coeff[:, :1]),
            torch.ones_like(coeff[:, :1]),
        )
        return energy, coeff * sign

    def forward(
        self,
        v: torch.Tensor,
        q_context: torch.Tensor,
        q_target: torch.Tensor,
        alpha: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> KrylovJEPAOutput:
        qs = [q_context[:, i] for i in range(self.context_steps)]
        cls, tokens, v_cls = self.fuse_context(v, qs)
        z_pred = self.predictor(cls, tokens)
        with torch.no_grad():
            z_tgt, _ = self.target_state(q_target)
            if self.normalize_latents:
                z_tgt = nn.functional.layer_norm(z_tgt, (z_tgt.shape[-1],))
        z_pred_n = (
            nn.functional.layer_norm(z_pred, (z_pred.shape[-1],))
            if self.normalize_latents
            else z_pred
        )
        loss_jepa = torch.mean((z_pred_n - z_tgt.detach()) ** 2)

        loss_coeff = torch.zeros((), device=v.device, dtype=v.dtype)
        if self.lambda_coeff > 0 and alpha is not None and beta is not None:
            j = self.context_steps - 1
            coeff_true = torch.stack([alpha[:, j], beta[:, j]], dim=-1)
            coeff_hat = self.coeff_head(cls)
            loss_coeff = torch.mean((coeff_hat - coeff_true) ** 2)

        loss_projected_ritz = torch.zeros((), device=v.device, dtype=v.dtype)
        if self.lambda_projected_ritz > 0:
            if v_cls is None:
                raise ValueError("projected-Ritz auxiliary task requires the potential input")
            with torch.no_grad():
                e_target, c_target = self._projected_ritz_targets(v)
            e_pred = self.ritz_energy_head(v_cls).squeeze(-1)
            c_pred = self.ritz_coeff_head(v_cls)
            e_scale = e_target.abs().clamp_min(1.0)
            loss_e = torch.mean(((e_pred - e_target) / e_scale) ** 2)
            c_pred = nn.functional.normalize(c_pred, dim=-1)
            c_target = nn.functional.normalize(c_target, dim=-1)
            overlap = torch.sum(c_pred * c_target, dim=-1)
            loss_c = torch.mean(1.0 - overlap.pow(2))
            loss_projected_ritz = loss_e + loss_c

        loss = (
            loss_jepa
            + self.lambda_coeff * loss_coeff
            + self.lambda_projected_ritz * loss_projected_ritz
        )
        return KrylovJEPAOutput(
            loss=loss,
            loss_jepa=loss_jepa,
            loss_coeff=loss_coeff,
            loss_projected_ritz=loss_projected_ritz,
            z_pred=z_pred_n,
            z_target=z_tgt,
        )

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.state_enc, self.target_state, self.ema_momentum)

    def potential_encoder(self) -> PotentialEncoder:
        return self.potential_enc
