"""Model package."""

from spectral_krylov_jepa.models.downstream import DownstreamGroundStateModel
from spectral_krylov_jepa.models.encoders import PotentialEncoder, StateEncoder, count_parameters
from spectral_krylov_jepa.models.field_jepa import FieldJEPA
from spectral_krylov_jepa.models.krylov_jepa import KrylovJEPA
from spectral_krylov_jepa.models.operator_jepa import OperatorJEPA

__all__ = [
    "PotentialEncoder",
    "StateEncoder",
    "FieldJEPA",
    "OperatorJEPA",
    "KrylovJEPA",
    "DownstreamGroundStateModel",
    "count_parameters",
]
