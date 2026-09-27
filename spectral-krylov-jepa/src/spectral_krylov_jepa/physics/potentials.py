"""Reproducible random potential families for ID and OOD evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec

FamilyName = Literal[
    "id_gaussian_mixture",
    "ood_narrow",
    "ood_strong",
    "ood_double",
    "ood_rough",
    "box_zero",
]


@dataclass
class WellParams:
    amplitude: float
    sigma: float
    cx: float
    cy: float


@dataclass
class PotentialSpec:
    """Fully specified potential instance for reproducibility."""

    family: FamilyName
    seed: int
    grid: GridSpec
    wells: list[WellParams] = field(default_factory=list)
    rough_amplitude: float = 0.0
    rough_modes: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": int(self.seed),
            "grid": self.grid.to_dict(),
            "wells": [asdict(w) for w in self.wells],
            "rough_amplitude": float(self.rough_amplitude),
            "rough_modes": int(self.rough_modes),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PotentialSpec":
        wells = [WellParams(**w) for w in d.get("wells", [])]
        return cls(
            family=d["family"],
            seed=int(d["seed"]),
            grid=GridSpec.from_dict(d["grid"]),
            wells=wells,
            rough_amplitude=float(d.get("rough_amplitude", 0.0)),
            rough_modes=int(d.get("rough_modes", 0)),
            notes=str(d.get("notes", "")),
        )


@dataclass(frozen=True)
class FamilyConfig:
    """Sampling ranges for a potential family."""

    name: FamilyName
    n_wells_min: int
    n_wells_max: int
    amplitude_min: float
    amplitude_max: float
    sigma_min: float
    sigma_max: float
    margin: float = 0.15
    min_separation: float = 0.0
    rough_amplitude: float = 0.0
    rough_modes: int = 0


# Frozen ID / OOD ranges (see paper/experiment_protocol.md)
ID_FAMILY = FamilyConfig(
    name="id_gaussian_mixture",
    n_wells_min=1,
    n_wells_max=3,
    amplitude_min=-8.0,
    amplitude_max=-2.0,
    sigma_min=0.08,
    sigma_max=0.18,
    margin=0.15,
    min_separation=0.0,
)

OOD_NARROW = FamilyConfig(
    name="ood_narrow",
    n_wells_min=1,
    n_wells_max=2,
    amplitude_min=-8.0,
    amplitude_max=-2.0,
    sigma_min=0.035,
    sigma_max=0.055,
    margin=0.15,
)

OOD_STRONG = FamilyConfig(
    name="ood_strong",
    n_wells_min=1,
    n_wells_max=2,
    amplitude_min=-18.0,
    amplitude_max=-12.0,
    sigma_min=0.08,
    sigma_max=0.18,
    margin=0.15,
)

OOD_DOUBLE = FamilyConfig(
    name="ood_double",
    n_wells_min=2,
    n_wells_max=2,
    amplitude_min=-8.0,
    amplitude_max=-3.0,
    sigma_min=0.08,
    sigma_max=0.14,
    margin=0.12,
    min_separation=0.45,
)

OOD_ROUGH = FamilyConfig(
    name="ood_rough",
    n_wells_min=1,
    n_wells_max=2,
    amplitude_min=-6.0,
    amplitude_max=-2.0,
    sigma_min=0.10,
    sigma_max=0.18,
    margin=0.15,
    rough_amplitude=1.5,
    rough_modes=4,
)

FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    cfg.name: cfg
    for cfg in (ID_FAMILY, OOD_NARROW, OOD_STRONG, OOD_DOUBLE, OOD_ROUGH)
}


def _sample_wells(rng: np.random.Generator, cfg: FamilyConfig, grid: GridSpec) -> list[WellParams]:
    n_wells = int(rng.integers(cfg.n_wells_min, cfg.n_wells_max + 1))
    wells: list[WellParams] = []
    xmin = grid.x_min + cfg.margin * (grid.x_max - grid.x_min)
    xmax = grid.x_max - cfg.margin * (grid.x_max - grid.x_min)
    ymin = grid.y_min + cfg.margin * (grid.y_max - grid.y_min)
    ymax = grid.y_max - cfg.margin * (grid.y_max - grid.y_min)

    attempts = 0
    while len(wells) < n_wells and attempts < 5000:
        attempts += 1
        cx = float(rng.uniform(xmin, xmax))
        cy = float(rng.uniform(ymin, ymax))
        if cfg.min_separation > 0 and wells:
            dists = [np.hypot(cx - w.cx, cy - w.cy) for w in wells]
            if min(dists) < cfg.min_separation:
                continue
        amp = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
        sigma = float(rng.uniform(cfg.sigma_min, cfg.sigma_max))
        wells.append(WellParams(amplitude=amp, sigma=sigma, cx=cx, cy=cy))
    if len(wells) < n_wells:
        raise RuntimeError(
            f"Failed to sample {n_wells} wells under constraints for family={cfg.name}"
        )
    return wells


def evaluate_potential(spec: PotentialSpec) -> np.ndarray:
    """Evaluate V(x,y) on the interior grid."""
    grid = spec.grid
    X, Y = grid.meshgrid()
    v = np.zeros_like(X, dtype=np.float64)
    for w in spec.wells:
        r2 = (X - w.cx) ** 2 + (Y - w.cy) ** 2
        v += w.amplitude * np.exp(-r2 / (2.0 * w.sigma**2))

    if spec.rough_amplitude > 0 and spec.rough_modes > 0:
        # Deterministic high-frequency perturbation from the same seed branch
        rng = np.random.default_rng(spec.seed + 17_389)
        for _ in range(spec.rough_modes):
            kx = int(rng.integers(3, 8))
            ky = int(rng.integers(3, 8))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            amp = float(rng.uniform(-spec.rough_amplitude, spec.rough_amplitude))
            v += amp * np.sin(2.0 * np.pi * kx * X + phase) * np.sin(2.0 * np.pi * ky * Y)

    return v


def generate_potential(
    family: FamilyName | str,
    seed: int,
    grid: GridSpec | None = None,
) -> tuple[np.ndarray, PotentialSpec]:
    """Generate a potential field and its full specification."""
    grid = grid or GridSpec()
    if family == "box_zero":
        spec = PotentialSpec(family="box_zero", seed=seed, grid=grid, wells=[], notes="V=0 box")
        return evaluate_potential(spec), spec

    if family not in FAMILY_CONFIGS:
        raise ValueError(f"Unknown family {family!r}. Known: {list(FAMILY_CONFIGS)}")

    cfg = FAMILY_CONFIGS[family]
    rng = np.random.default_rng(seed)
    wells = _sample_wells(rng, cfg, grid)
    spec = PotentialSpec(
        family=cfg.name,  # type: ignore[arg-type]
        seed=seed,
        grid=grid,
        wells=wells,
        rough_amplitude=cfg.rough_amplitude,
        rough_modes=cfg.rough_modes,
    )
    return evaluate_potential(spec), spec


def generate_box_potential(grid: GridSpec | None = None) -> tuple[np.ndarray, PotentialSpec]:
    """Infinite-box-like V = 0 on the interior (Dirichlet boundary separate)."""
    return generate_potential("box_zero", seed=0, grid=grid or GridSpec())
