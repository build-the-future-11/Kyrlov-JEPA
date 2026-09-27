"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer(
    params: Iterable[nn.Parameter] | nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    if isinstance(params, nn.Module):
        params = [p for p in params.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if total_steps <= 0:
        return None
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    if warmup_steps > 0:
        warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=total_steps)


def optimizer_to_dict(opt: torch.optim.Optimizer) -> dict[str, Any]:
    return {"type": opt.__class__.__name__, "param_groups": [
        {k: v for k, v in g.items() if k != "params"} for g in opt.param_groups
    ]}
