"""Shared FNO-2D backbone, downstream solver, LMOP-JEPA."""

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


class LMOPJEPA(nn.Module):
    """Same-arch context/target encoders (3-channel). Target is EMA copy.

    Context input: (a, f⊙(1-M), M)
    Target input:  (u, u, 0)  — solution view, no forcing leak
    """

    def __init__(
        self,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
        ema_momentum: float = 0.996,
        mask_ratio: float = 0.4,
        var_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.mask_ratio = mask_ratio
        self.var_loss_weight = var_loss_weight
        self.width = width
        self.online = FNO2d(3, 1, width=width, modes=modes, n_layers=n_layers)
        self.target = freeze_module(copy.deepcopy(self.online))
        self.predictor = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def random_mask(self, b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        n = h * w
        n_mask = max(1, int(round(n * self.mask_ratio)))
        noise = torch.rand(b, n, device=device)
        idx = torch.argsort(noise, dim=1)[:, :n_mask]
        mask = torch.zeros(b, n, device=device)
        mask.scatter_(1, idx, 1.0)
        return mask.view(b, h, w)

    @staticmethod
    def _ln(z: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(z.permute(0, 2, 3, 1), (z.shape[1],)).permute(0, 3, 1, 2)

    def forward(self, a: torch.Tensor, f: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, h, w = a.shape
        mask = self.random_mask(b, h, w, a.device)
        x_c = torch.stack([a, f * (1.0 - mask), mask], dim=1)
        zeros = torch.zeros_like(u)
        x_t = torch.stack([u, u, zeros], dim=1)
        z_c = self.online.encode(x_c)
        z_pred = self._ln(self.predictor(z_c))
        with torch.no_grad():
            z_t = self._ln(self.target.encode(x_t))
        loss_jepa = torch.mean((z_pred - z_t) ** 2)
        # VICReg-style variance hinge (Amendment 0002)
        flat = z_pred.reshape(b, z_pred.shape[1], -1).mean(-1)  # B, C
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        loss_var = torch.mean(torch.relu(1.0 - std))
        loss = loss_jepa + float(self.var_loss_weight) * loss_var
        with torch.no_grad():
            var = float(std.mean().item() ** 2)
            norms = flat.norm(dim=1).mean().item()
        return loss, {
            "latent_var": var,
            "latent_std": float(std.mean().item()),
            "latent_norm": float(norms),
            "loss_jepa": float(loss_jepa.item()),
            "loss_var": float(loss_var.item()),
        }

    @torch.no_grad()
    def update_target(self) -> None:
        update_ema(self.online, self.target, self.ema_momentum)

    def transfer_to_downstream(self, dest: DownstreamSolver) -> None:
        with torch.no_grad():
            # Copy spectral blocks; lift partially from online.lift (first 2 of 3 in-ch)
            for d, s in zip(dest.backbone.blocks, self.online.blocks, strict=True):
                d.load_state_dict(s.state_dict())
            # Approximate lift: average first two input channels weights for (a,f)
            w = self.online.lift.weight  # (width, 3, 1, 1)
            dest.backbone.lift.weight.copy_(w[:, :2].contiguous())
            if self.online.lift.bias is not None and dest.backbone.lift.bias is not None:
                dest.backbone.lift.bias.copy_(self.online.lift.bias)
