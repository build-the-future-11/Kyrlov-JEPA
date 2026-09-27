"""Ablation experiment helpers."""

from __future__ import annotations

from typing import Any


ABLATION_SPECS: dict[str, dict[str, Any]] = {
    "krylov_k1": {"method": "krylov", "context_steps": 1, "lambda_coeff": 0.0},
    "krylov_k2": {"method": "krylov", "context_steps": 2, "lambda_coeff": 0.0},
    "krylov_k3": {"method": "krylov", "context_steps": 3, "lambda_coeff": 0.0},
    "krylov_coeff": {"method": "krylov", "context_steps": 2, "lambda_coeff": 0.1},
    "krylov_remove_v": {"method": "krylov", "context_steps": 2, "remove_v": True},
    "krylov_shuffled": {"method": "krylov", "context_steps": 2, "shuffle_physics": True},
    "operator": {"method": "operator"},
    "field": {"method": "field"},
}


def list_ablations() -> list[str]:
    return list(ABLATION_SPECS.keys())


def get_ablation_config(name: str) -> dict[str, Any]:
    if name not in ABLATION_SPECS:
        raise KeyError(f"Unknown ablation {name!r}. Known: {list_ablations()}")
    return dict(ABLATION_SPECS[name])
