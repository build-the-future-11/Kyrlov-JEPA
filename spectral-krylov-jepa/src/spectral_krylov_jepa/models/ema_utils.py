"""EMA target encoder utilities."""

from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn


def clone_as_ema(module: nn.Module) -> nn.Module:
    """Deep-copy a module and freeze parameters for EMA target use."""
    target = copy.deepcopy(module)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """Exponential moving average update: target = m * target + (1-m) * online."""
    if not (0.0 <= momentum <= 1.0):
        raise ValueError(f"momentum must be in [0,1], got {momentum}")
    for p_online, p_target in zip(online.parameters(), target.parameters(), strict=True):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)
    for b_online, b_target in zip(online.buffers(), target.buffers(), strict=True):
        b_target.data.copy_(b_online.data)


def set_requires_grad(modules: Iterable[nn.Module], requires_grad: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(requires_grad)
