"""Re-export frozen Darcy reference."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from darcy_reference import (  # noqa: E402
    DarcyGrid,
    apply_operator,
    build_darcy_matrix,
    condition_estimate,
    is_spd_probe,
    manufactured_round_trip,
    manufactured_sine_mode,
    matrix_symmetry_error,
    sample_log_permeability,
    solve_darcy,
    validate_permeability,
)

__all__ = [
    "DarcyGrid",
    "apply_operator",
    "build_darcy_matrix",
    "condition_estimate",
    "is_spd_probe",
    "manufactured_round_trip",
    "manufactured_sine_mode",
    "matrix_symmetry_error",
    "sample_log_permeability",
    "solve_darcy",
    "validate_permeability",
]
