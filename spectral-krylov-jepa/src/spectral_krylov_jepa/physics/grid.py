"""Rectangular grid utilities for 2D finite-difference Schrödinger problems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Uniform rectangular grid on [x_min, x_max] × [y_min, y_max].

    Interior nodes exclude the Dirichlet boundary. For ``n_interior=32`` on
    the unit square, spacing is ``h = 1 / (n_interior + 1)``.
    """

    n_interior: int = 32
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def __post_init__(self) -> None:
        if self.n_interior < 2:
            raise ValueError(f"n_interior must be >= 2, got {self.n_interior}")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Grid extents must satisfy max > min")

    @property
    def nx(self) -> int:
        return self.n_interior

    @property
    def ny(self) -> int:
        return self.n_interior

    @property
    def n_dof(self) -> int:
        """Number of free interior degrees of freedom."""
        return self.nx * self.ny

    @property
    def hx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx + 1)

    @property
    def hy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny + 1)

    @property
    def h(self) -> float:
        """Isotropic spacing (requires square cells)."""
        if not np.isclose(self.hx, self.hy):
            raise ValueError(f"Non-square cells: hx={self.hx}, hy={self.hy}")
        return float(self.hx)

    def x_coords(self) -> np.ndarray:
        return self.x_min + self.hx * np.arange(1, self.nx + 1, dtype=np.float64)

    def y_coords(self) -> np.ndarray:
        return self.y_min + self.hy * np.arange(1, self.ny + 1, dtype=np.float64)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) with shape (ny, nx), row-major y then x."""
        x = self.x_coords()
        y = self.y_coords()
        return np.meshgrid(x, y, indexing="xy")

    def flatten_field(self, field: np.ndarray) -> np.ndarray:
        """Flatten a (ny, nx) field to length-n_dof vector (row-major)."""
        arr = np.asarray(field, dtype=np.float64)
        if arr.shape != (self.ny, self.nx):
            raise ValueError(f"Expected shape {(self.ny, self.nx)}, got {arr.shape}")
        return arr.reshape(-1)

    def reshape_vector(self, vec: np.ndarray) -> np.ndarray:
        """Reshape length-n_dof vector to (ny, nx)."""
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size != self.n_dof:
            raise ValueError(f"Expected length {self.n_dof}, got {arr.size}")
        return arr.reshape(self.ny, self.nx)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GridSpec":
        return cls(
            n_interior=int(d.get("n_interior", d.get("nx", 32))),
            x_min=float(d.get("x_min", 0.0)),
            x_max=float(d.get("x_max", 1.0)),
            y_min=float(d.get("y_min", 0.0)),
            y_max=float(d.get("y_max", 1.0)),
        )


def cell_area(grid: GridSpec) -> float:
    """Area element for discrete L2 inner products."""
    return float(grid.hx * grid.hy)
