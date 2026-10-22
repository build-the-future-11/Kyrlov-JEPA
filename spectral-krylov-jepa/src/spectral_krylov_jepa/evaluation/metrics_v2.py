"""Strict, opt-in development evaluation; the retained metrics module is unchanged.

Run ``python -m spectral_krylov_jepa.evaluation.metrics_v2 INPUT.json RECEIPT.json``.
See research/DEVELOPMENT_STRICT_METRICS_20261010.md for the versioned contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from spectral_krylov_jepa.evaluation.metrics import rayleigh_quotient, schrodinger_residual
from spectral_krylov_jepa.physics.grid import GridSpec

CONTRACT = "spectral-metrics-v2"
NORM_TOLERANCE = 1e-6
METRIC_KEYS = (
    "fidelity", "infidelity", "rel_energy_error", "rel_rayleigh_energy_error",
    "rayleigh_energy", "residual_rel", "residual_true_e", "residual_rayleigh",
    "sign_aligned_rel_l2",
)
CASE_KEYS = {"case_id", "potential", "psi_true", "e_true", "psi_hat", "e_hat"}
GRID_KEYS = {"n_interior", "x_min", "x_max", "y_min", "y_max"}


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as float64") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, Integral) and int(result) != int(value):
        raise ValueError(f"{name} is not exactly representable as float64")
    if isinstance(value, np.floating) and value.dtype.itemsize > 8 and value != result:
        raise ValueError(f"{name} is not exactly representable as float64")
    return result


def _grid_area(grid: GridSpec) -> float:
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec")
    if isinstance(grid.n_interior, (bool, np.bool_)) or not isinstance(
        grid.n_interior, Integral
    ) or grid.n_interior < 2:
        raise ValueError("n_interior must be an integer >= 2")
    for key in GRID_KEYS - {"n_interior"}:
        _scalar(getattr(grid, key), key)
    for name, spacing in (("hx", grid.hx), ("hy", grid.hy)):
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"{name} must be finite and positive")
        square = spacing * spacing
        if not math.isfinite(square) or square <= 0:
            raise ValueError(f"{name} squared is not representable")
        inverse = 1.0 / square
        if not math.isfinite(inverse) or inverse <= 0:
            raise ValueError(f"inverse {name} squared is not representable")
    area = grid.hx * grid.hy
    if not math.isfinite(area) or area <= 0:
        raise ValueError("cell area must be finite and positive")
    return area


def _field(value: Any, name: str, grid: GridSpec) -> np.ndarray:
    try:
        # Object conversion preserves Python integers and booleans in mixed JSON
        # lists until admission. Inferring a float dtype first would erase them.
        raw = value if isinstance(value, np.ndarray) else np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real array") from exc
    if raw.shape != (grid.ny, grid.nx) or raw.dtype.kind not in "fiuO":
        raise ValueError(f"{name} must be a real array of shape {(grid.ny, grid.nx)}")
    # Integer coercion must not erase a represented prediction/reference error.
    if raw.dtype.kind in "iuO" or raw.dtype.itemsize > 8:
        for item in raw.flat:
            _scalar(item, name)
    with np.errstate(over="ignore", invalid="ignore"):
        array = raw.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def _unit_state(state: np.ndarray, area: float, name: str) -> np.ndarray:
    flat = state.reshape(-1)
    scale = float(np.max(np.abs(flat)))
    if scale == 0:
        raise ValueError(f"{name} is a zero state")
    scaled = flat / scale
    norm = math.hypot(*scaled)
    log_norm = math.log(scale) + math.log(norm) + 0.5 * math.log(area)
    if not math.log1p(-NORM_TOLERANCE) <= log_norm <= math.log1p(NORM_TOLERANCE):
        raise ValueError(f"{name} physical L2 norm must be within {NORM_TOLERANCE} of 1")
    return scaled / norm


def evaluate_example(
    potential: np.ndarray,
    psi_true: np.ndarray,
    e_true: float,
    psi_hat: np.ndarray,
    e_hat: float,
    grid: GridSpec,
) -> dict[str, float]:
    """Evaluate an admitted pair; invalid states raise instead of receiving scores.

    Reference correctness is supplied by the caller, not inferred by this API.
    The three residuals use the same Euclidean unit prediction and operator.
    """
    area = _grid_area(grid)
    potential = _field(potential, "potential", grid)
    reference = _unit_state(_field(psi_true, "psi_true", grid), area, "psi_true")
    prediction = _unit_state(_field(psi_hat, "psi_hat", grid), area, "psi_hat")
    e_true, e_hat = _scalar(e_true, "e_true"), _scalar(e_hat, "e_hat")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # Reuse the independently developed amplitude repair from dependency #22.
        # This successor supplies state admission and complete-cohort accounting.
        rayleigh = rayleigh_quotient(potential, prediction, grid)
        overlap = math.fsum(float(x) * float(y) for x, y in zip(prediction, reference))
        # Both admitted vectors have unit Euclidean norm. Clip only dot-product roundoff.
        fidelity = min(1.0, abs(overlap)) ** 2
        aligned = -prediction if overlap < 0 else prediction
        result = {
            "fidelity": fidelity,
            "infidelity": 1.0 - fidelity,
            "rel_energy_error": abs(e_hat - e_true) / (abs(e_true) + 1e-6),
            "rel_rayleigh_energy_error": abs(rayleigh - e_true) / (abs(e_true) + 1e-6),
            "rayleigh_energy": rayleigh,
            "residual_rel": schrodinger_residual(potential, prediction, e_hat, grid),
            "residual_true_e": schrodinger_residual(potential, prediction, e_true, grid),
            "residual_rayleigh": schrodinger_residual(potential, prediction, rayleigh, grid),
            "sign_aligned_rel_l2": math.hypot(*(aligned - reference)),
        }
    return {key: _scalar(value, key) for key, value in result.items()}


def _moments(values: list[float]) -> tuple[float, float]:
    # Exact represented-value accumulation handles huge sums and genuine signed
    # cancellation. Reject a nonzero result that float64 would erase to zero.
    rational = [Fraction.from_float(value) for value in values]
    exact = sum(rational, Fraction()) / len(values)
    mean = float(exact)
    if not math.isfinite(mean) or (mean == 0 and exact != 0):
        raise ValueError("metric mean is not representable as a nonzero finite float64")
    variance = sum(((value - exact) ** 2 for value in rational), Fraction()) / len(values)
    if variance == 0:
        return mean, 0.0
    # Center on the exact represented-value mean, before float rounding. This
    # matters for adjacent floats whose exact midpoint is not representable.
    scale = max(abs(value) for value in values)
    scaled_variance = float(variance / Fraction.from_float(scale) ** 2)
    std = math.sqrt(scaled_variance) * scale
    if not math.isfinite(std) or std == 0:
        raise ValueError("metric standard deviation is not representable")
    return mean, std


def summarize_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Population mean/std and raw values for the complete finite cohort only."""
    if not rows:
        raise ValueError("a summary requires at least one metric row")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(METRIC_KEYS):
            raise ValueError("each metric row must contain exactly the declared metric keys")
    result: dict[str, Any] = {"n": len(rows)}
    for key in METRIC_KEYS:
        values = [_scalar(row[key], key) for row in rows]
        mean, std = _moments(values)
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = std
        result[f"{key}_values"] = values
    return result


def audit_examples(cases: list[dict[str, Any]], grid: GridSpec) -> dict[str, Any]:
    """Keep invalid case records and withhold aggregation for an incomplete cohort."""
    _grid_area(grid)
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ValueError("each case must contain exactly the declared case fields")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("case_id must be a unique nonempty string")
        ids.add(case_id)
    records = []
    for case in cases:
        try:
            metrics = evaluate_example(
                case["potential"], case["psi_true"], case["e_true"],
                case["psi_hat"], case["e_hat"], grid,
            )
            records.append({"case_id": case["case_id"], "status": "VALID", "metrics": metrics})
        except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            records.append({"case_id": case["case_id"], "status": "INVALID", "error": str(exc)})
    invalid = sum(record["status"] == "INVALID" for record in records)
    result = {
        "contract": CONTRACT, "evaluation_mode": "development",
        "status": "INVALID" if invalid else "VALID",
        "n_requested": len(cases), "n_valid": len(cases) - invalid, "n_invalid": invalid,
        "cases": records, "summary": None,
    }
    if not invalid:
        try:
            result["summary"] = summarize_metrics([record["metrics"] for record in records])
        except (ValueError, OverflowError) as exc:
            result.update(status="INVALID", summary_error=str(exc))
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    # Reserve first: an existing result is never truncated, even after a failure.
    try:
        output = args.receipt.open("x", encoding="utf-8")
    except OSError as exc:
        print(f"cannot reserve receipt: {exc}", file=sys.stderr)
        return 2
    with output:
        package = Path(__file__).resolve().parents[1]
        source_paths = [Path(__file__), package / "evaluation" / "metrics.py",
                        *(package / "physics" / name for name in (
            "grid.py", "hamiltonian.py", "laplacian.py",
        ))]
        report: dict[str, Any] = {
            "contract": CONTRACT, "status": "INVALID", "summary": None,
            "source_sha256": {str(path.relative_to(package)): hashlib.sha256(
                path.read_bytes()).hexdigest() for path in source_paths},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "platform": platform.platform()},
        }
        try:
            raw = args.input.read_bytes()
            report["input_sha256"] = hashlib.sha256(raw).hexdigest()
            report["input_base64"] = base64.b64encode(raw).decode("ascii")
            payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                                 object_pairs_hook=_unique_object)
            if not isinstance(payload, dict) or set(payload) != {"evaluation_mode", "grid", "cases"}:
                raise ValueError("input must contain exactly evaluation_mode, grid and cases")
            if payload["evaluation_mode"] != "development":
                raise ValueError("this successor admits development evaluation only")
            if not isinstance(payload["grid"], dict) or set(payload["grid"]) != GRID_KEYS:
                raise ValueError("grid must explicitly declare n_interior and all four extents")
            grid = GridSpec(**payload["grid"])
            report.update(audit_examples(payload["cases"], grid))
        except (OSError, UnicodeError, ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            report["error"] = str(exc)
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if report["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
