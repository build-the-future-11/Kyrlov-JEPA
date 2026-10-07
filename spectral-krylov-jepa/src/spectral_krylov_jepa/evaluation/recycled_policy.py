"""Development-only policy freeze for the recycled Ritz control.

The functions in this module never inspect protected labels or protected solver
outcomes. They consume a ledger of normalized full-space Ritz residuals from
explicitly development-only cases and turn it into a deterministic rank and
refresh-threshold policy.

This is intentionally separate from result-bearing evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DevelopmentResidual:
    """One development-only residual observation for one candidate rank."""

    case_id: str
    rank: int
    residual_relative_to_hx: float
    operator_applications: int
    source_role: str = "development"


@dataclass(frozen=True)
class RecycledRitzPolicy:
    """Frozen policy produced without protected-outcome access."""

    rank: int
    refresh_threshold: float
    ranking_quantile: float
    threshold_quantile: float
    near_best_factor: float
    max_operator_applications: int
    n_development_cases: int
    candidate_ranks: tuple[int, ...]
    ranking_residuals: tuple[tuple[int, float], ...]
    selection_rule: str = (
        "choose the smallest budget-legal rank whose development residual "
        "quantile is within near_best_factor of the best legal rank; then "
        "freeze the refresh threshold as the conservative development residual quantile"
    )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["candidate_ranks"] = list(self.candidate_ranks)
        payload["ranking_residuals"] = [
            {"rank": rank, "residual_quantile": residual}
            for rank, residual in self.ranking_residuals
        ]
        return payload


def _higher_quantile(values: Sequence[float], q: float) -> float:
    """Conservative empirical quantile with no interpolation below observations."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantile input must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("residuals must be finite and non-negative")
    return float(np.quantile(array, q, method="higher"))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer without coercion")
    return int(value)


def development_records_from_payload(payload: object) -> list[DevelopmentResidual]:
    """Read declarations without silently coercing or relabeling evidence.

    Valid declarations do not prove the external provenance of a ledger. They
    are necessary input checks before the separately reviewed source identity.
    """
    if not isinstance(payload, dict) or payload.get("source_role") != "development":
        raise ValueError("development ledger must declare source_role=development")
    for field in ("protected_outcomes_opened", "query_exact_eigensolves_performed"):
        if payload.get(field) is not False:
            raise ValueError(f"development ledger must explicitly declare {field}=false")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("development ledger must contain a nonempty records list")
    required = {"case_id", "rank", "residual_relative_to_hx", "operator_applications"}
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"record {index} is missing required ledger fields")
        records.append(DevelopmentResidual(
            case_id=row["case_id"],
            rank=row["rank"],
            residual_relative_to_hx=row["residual_relative_to_hx"],
            operator_applications=row["operator_applications"],
            source_role=row.get("source_role", payload["source_role"]),
        ))
    return records


def freeze_recycled_policy(
    records: Sequence[DevelopmentResidual],
    *,
    candidate_ranks: Sequence[int],
    ranking_quantile: float = 0.90,
    threshold_quantile: float = 0.95,
    near_best_factor: float = 1.10,
    max_operator_applications: int | None = None,
) -> RecycledRitzPolicy:
    """Freeze rank and refresh threshold from development-only residuals.

    The ranking rule intentionally uses only a test-time-available diagnostic:
    the normalized full-space Ritz residual. It never uses protected energies,
    eigenvectors, labels, or success/failure outcomes.

    Candidate ranks and all scalar rule parameters must be declared before
    protected evaluation. Every rank must be evaluated on the identical set of
    development case IDs.
    """
    if not records:
        raise ValueError("at least one development residual record is required")
    if not candidate_ranks:
        raise ValueError("candidate_ranks must be nonempty")
    if isinstance(ranking_quantile, bool) or not isinstance(ranking_quantile, Real) or not 0 < ranking_quantile <= 1:
        raise ValueError("ranking_quantile must be in (0, 1]")
    if isinstance(threshold_quantile, bool) or not isinstance(threshold_quantile, Real) or not 0 < threshold_quantile <= 1:
        raise ValueError("threshold_quantile must be in (0, 1]")
    if isinstance(near_best_factor, bool) or not isinstance(near_best_factor, Real) or not np.isfinite(near_best_factor) or near_best_factor < 1:
        raise ValueError("near_best_factor must be finite and at least 1")

    ranks = tuple(sorted(set(_positive_integer(rank, "candidate rank") for rank in candidate_ranks)))
    if len(ranks) != len(candidate_ranks):
        raise ValueError("candidate ranks must be unique")

    if max_operator_applications is None:
        max_operator_applications = max(ranks)
    max_operator_applications = _positive_integer(max_operator_applications, "max_operator_applications")

    by_rank: dict[int, list[DevelopmentResidual]] = {rank: [] for rank in ranks}
    for record in records:
        if record.source_role != "development":
            raise ValueError(
                "policy freeze accepts development-only records; got source_role={!r}".format(
                    record.source_role
                )
            )
        if not isinstance(record.case_id, str) or not record.case_id.strip():
            raise ValueError("case_id must be nonempty")
        _positive_integer(record.rank, "record rank")
        _positive_integer(record.operator_applications, "operator_applications")
        if record.rank not in by_rank:
            raise ValueError("record rank {} was not preregistered".format(record.rank))
        if (
            isinstance(record.residual_relative_to_hx, bool)
            or not isinstance(record.residual_relative_to_hx, Real)
            or not np.isfinite(record.residual_relative_to_hx)
            or record.residual_relative_to_hx < 0
        ):
            raise ValueError("residual_relative_to_hx must be finite and non-negative")
        if record.operator_applications != record.rank:
            raise ValueError(
                "operator application ledger must equal recycled basis rank before fallback"
            )
        by_rank[record.rank].append(record)

    expected_case_ids: set[str] | None = None
    for rank in ranks:
        items = by_rank[rank]
        if not items:
            raise ValueError("candidate rank {} has no development records".format(rank))
        case_ids = [item.case_id for item in items]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate case_id for candidate rank {}".format(rank))
        current = set(case_ids)
        if expected_case_ids is None:
            expected_case_ids = current
        elif current != expected_case_ids:
            raise ValueError("every candidate rank must use the identical development cases")

    assert expected_case_ids is not None
    legal_ranks = [rank for rank in ranks if rank <= max_operator_applications]
    if not legal_ranks:
        raise ValueError("no candidate rank fits max_operator_applications")

    ranking = {
        rank: _higher_quantile(
            [item.residual_relative_to_hx for item in by_rank[rank]],
            ranking_quantile,
        )
        for rank in legal_ranks
    }
    best_residual = min(ranking.values())
    tolerance = best_residual * near_best_factor

    eligible = [
        rank
        for rank in legal_ranks
        if ranking[rank] <= tolerance + np.finfo(np.float64).eps
    ]
    chosen_rank = min(eligible)
    refresh_threshold = _higher_quantile(
        [item.residual_relative_to_hx for item in by_rank[chosen_rank]],
        threshold_quantile,
    )

    return RecycledRitzPolicy(
        rank=chosen_rank,
        refresh_threshold=refresh_threshold,
        ranking_quantile=float(ranking_quantile),
        threshold_quantile=float(threshold_quantile),
        near_best_factor=float(near_best_factor),
        max_operator_applications=int(max_operator_applications),
        n_development_cases=len(expected_case_ids),
        candidate_ranks=ranks,
        ranking_residuals=tuple((rank, ranking[rank]) for rank in legal_ranks),
    )
