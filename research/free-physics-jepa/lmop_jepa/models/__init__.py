"""Shared FNO-2d backbone, downstream solver, LMOP-JEPA v2 (amendment 0003)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, modes, 2))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = min(self.modes, h, w // 2 + 1)
        w_c = torch.view_as_complex(self.weight[:, :, :m, :m].contiguous())
        out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m, :m], w_c)
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.spec(x) + self.w(x))


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))
        self.width = width
        self.in_channels = in_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encode(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    for p_o, p_t in zip(online.parameters(), target.parameters(), strict=True):
        p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class DownstreamSolver(nn.Module):
    def __init__(self, width: int = 32, modes: int = 12, n_layers: int = 4) -> None:
        super().__init__()
        self.backbone = FNO2d(2, 1, width=width, modes=modes, n_layers=n_layers)

    def forward(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.stack([a, f], dim=1)).squeeze(1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.lift.parameters():
            p.requires_grad_(trainable)
        for blk in self.backbone.blocks:
            for p in blk.parameters():
                p.requires_grad_(trainable)
        # Always train projection head
        for p in self.backbone.proj.parameters():
            p.requires_grad_(True)


class LMOPJEPA(nn.Module):
    """Amendment 0003: same-encoder multi-block JEPA + VICReg + hybrid u head.

    Online encoder is identical to DownstreamSolver.backbone (2-channel FNO).
    Context: normalized (a, f) with multi-block spatial zero-mask.
    Target: EMA encoder on stacked (u_n, u_n); stop-grad; loss on masked sites.
    """

    def __init__(
        self,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        ema_momentum: float = 0.996,
        mask_ratio: float = 0.3,
        mask_blocks_min: int = 1,
        mask_blocks_max: int = 3,
        lambda_var: float = 1.0,
        lambda_cov: float = 1.0,
        lambda_u: float = 1.0,
        var_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.mask_ratio = mask_ratio
        self.mask_blocks_min = mask_blocks_min
        self.mask_blocks_max = mask_blocks_max
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_u = lambda_u
        self.var_gamma = var_gamma
        self.width = width
        self.online = FNO2d(2, 1, width=width, modes=modes, n_layers=n_layers)
        self.target = freeze_module(copy.deepcopy(self.online))
        self.predictor = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def multi_block_mask(self, b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Return (B,H,W) mask with 1 = target/masked site."""
        masks = torch.zeros(b, h, w, device=device)
        area = h * w
        for bi in range(b):
            n_blocks = int(torch.randint(self.mask_blocks_min, self.mask_blocks_max + 1, (1,)).item())
            target_area = max(1, int(round(area * self.mask_ratio)))
            per = max(1, target_area // n_blocks)
            filled = 0
            for _ in range(n_blocks):
                bh = max(1, int(round((per ** 0.5) * float(torch.empty(1).uniform_(0.7, 1.3).item()))))
                bw = max(1, int(round(per / bh)))
                bh = min(bh, h)
                bw = min(bw, w)
                top = int(torch.randint(0, max(1, h - bh + 1), (1,)).item())
                left = int(torch.randint(0, max(1, w - bw + 1), (1,)).item())
                masks[bi, top : top + bh, left : left + bw] = 1.0
                filled = int(masks[bi].sum().item())
                if filled >= target_area:
                    break
            if masks[bi].sum() < 1:
                masks[bi, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        return masks

    @staticmethod
    def _ln(z: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(z.permute(0, 2, 3, 1), (z.shape[1],)).permute(0, 3, 1, 2)

    def _vicreg(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        # z: (B,C,H,W) -> pool to (B,C)
        flat = z.reshape(z.shape[0], z.shape[1], -1).mean(-1)
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        loss_var = torch.mean(torch.relu(self.var_gamma - std))
        flat_c = flat - flat.mean(dim=0)
        cov = (flat_c.T @ flat_c) / max(1, flat.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        loss_cov = (off ** 2).sum() / flat.shape[1]
        return loss_var, loss_cov, float(std.mean().item())

    def forward(self, a: torch.Tensor, f: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, h, w = a.shape
        mask = self.multi_block_mask(b, h, w, a.device)
        x_full = torch.stack([a, f], dim=1)
        x_ctx = x_full * (1.0 - mask.unsqueeze(1))
        # Target view: normalized-solution stacked into 2 channels
        u_rms = torch.sqrt(torch.mean(u * u, dim=(1, 2), keepdim=True) + 1e-8)
        u_n = u / u_rms
        x_tgt = torch.stack([u_n, u_n], dim=1)

        z_c = self.online.encode(x_ctx)
        z_pred = self._ln(self.predictor(z_c))
        with torch.no_grad():
            z_t = self._ln(self.target.encode(x_tgt))

        m = mask.unsqueeze(1)
        denom = m.sum() * z_pred.shape[1] + 1e-8
        loss_jepa = ((z_pred - z_t) ** 2 * m).sum() / denom

        loss_var, loss_cov, latent_std = self._vicreg(z_pred)
        # Hybrid u head on unmasked full encode (uses online proj)
        u_hat = self.online(x_full).squeeze(1)
        diff = (u_hat - u).reshape(b, -1)
        loss_u = torch.mean(torch.sum(diff * diff, dim=1) / (torch.sum(u.reshape(b, -1) ** 2, dim=1) + 1e-8))

        loss = (
            loss_jepa
            + float(self.lambda_var) * loss_var
            + float(self.lambda_cov) * loss_cov
            + float(self.lambda_u) * loss_u
        )
        return loss, {
            "latent_std": latent_std,
            "loss_jepa": float(loss_jepa.item()),
            "loss_var": float(loss_var.item()),
            "loss_cov": float(loss_cov.item()),
            "loss_u": float(loss_u.item()),
            "mask_frac": float(mask.mean().item()),
        }

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.online, self.target, self.ema_momentum)

    def transfer_to_downstream(self, dest: DownstreamSolver) -> None:
        with torch.no_grad():
            dest.backbone.load_state_dict(self.online.state_dict(), strict=True)
