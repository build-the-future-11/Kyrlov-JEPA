"""Tests for development-only recycled Ritz policy freezing."""

import math

import pytest

from spectral_krylov_jepa.evaluation.recycled_policy import (
    DevelopmentResidual,
    freeze_recycled_policy,
)


def _records(rank: int, values: list[float], role: str = "development"):
    return [
        DevelopmentResidual(
            case_id="dev-{:02d}".format(index),
            rank=rank,
            residual_relative_to_hx=value,
            operator_applications=rank,
            source_role=role,
        )
        for index, value in enumerate(values)
    ]


def test_policy_chooses_smallest_rank_within_near_best_band():
    records = (
        _records(1, [0.30, 0.28, 0.25, 0.27, 0.29])
        + _records(2, [0.11, 0.10, 0.09, 0.10, 0.10])
        + _records(4, [0.10, 0.095, 0.09, 0.09, 0.09])
    )
    policy = freeze_recycled_policy(
        records,
        candidate_ranks=[1, 2, 4],
        ranking_quantile=0.80,
        threshold_quantile=1.0,
        near_best_factor=1.10,
        max_operator_applications=4,
    )
    assert policy.rank == 2
    assert policy.refresh_threshold == pytest.approx(0.11)
    assert policy.n_development_cases == 5
    assert policy.max_operator_applications == 4


def test_budget_excludes_more_expensive_rank_even_if_residual_is_lower():
    records = (
        _records(1, [0.30, 0.30, 0.30])
        + _records(2, [0.20, 0.20, 0.20])
        + _records(4, [0.01, 0.01, 0.01])
    )
    policy = freeze_recycled_policy(
        records,
        candidate_ranks=[1, 2, 4],
        ranking_quantile=1.0,
        threshold_quantile=1.0,
        near_best_factor=1.0,
        max_operator_applications=2,
    )
    assert policy.rank == 2
    assert [rank for rank, _ in policy.ranking_residuals] == [1, 2]


def test_protected_or_nondevelopment_record_is_rejected():
    records = _records(1, [0.1, 0.2])
    records[0] = DevelopmentResidual(
        case_id="protected-00",
        rank=1,
        residual_relative_to_hx=0.1,
        operator_applications=1,
        source_role="confirmatory",
    )
    with pytest.raises(ValueError, match="development-only"):
        freeze_recycled_policy(records, candidate_ranks=[1])


def test_all_ranks_must_use_identical_development_case_ids():
    records = _records(1, [0.1, 0.2]) + [
        DevelopmentResidual("dev-00", 2, 0.05, 2),
        DevelopmentResidual("different-case", 2, 0.04, 2),
    ]
    with pytest.raises(ValueError, match="identical development cases"):
        freeze_recycled_policy(records, candidate_ranks=[1, 2])


def test_operator_application_ledger_must_match_rank_before_fallback():
    record = DevelopmentResidual(
        case_id="dev-00",
        rank=2,
        residual_relative_to_hx=0.1,
        operator_applications=1,
    )
    with pytest.raises(ValueError, match="operator application ledger"):
        freeze_recycled_policy([record], candidate_ranks=[2])


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_residuals_fail_closed(bad):
    record = DevelopmentResidual("dev-00", 1, bad, 1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        freeze_recycled_policy([record], candidate_ranks=[1])


def test_record_order_does_not_change_policy():
    records = (
        _records(1, [0.25, 0.20, 0.30, 0.22])
        + _records(2, [0.10, 0.12, 0.11, 0.09])
    )
    forward = freeze_recycled_policy(
        records,
        candidate_ranks=[1, 2],
        ranking_quantile=0.75,
        threshold_quantile=1.0,
        near_best_factor=1.05,
    )
    reverse = freeze_recycled_policy(
        list(reversed(records)),
        candidate_ranks=[1, 2],
        ranking_quantile=0.75,
        threshold_quantile=1.0,
        near_best_factor=1.05,
    )
    assert forward == reverse
