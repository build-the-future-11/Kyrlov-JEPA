"""Claim-gate logic for LMOP-JEPA (degenerate seed CIs must never count as support)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "research" / "free-physics-jepa"
sys.path.insert(0, str(PKG / "scripts"))

import aggregate_claim_gate as gate  # noqa: E402

PROTO = {"seeds": [11, 23, 47], "data_sizes": {"confirmatory": {"subsets": [25, 100]}}, "amendments": []}


def _rows(values: dict[str, list[float]]) -> list[dict]:
    rows = []
    for method, per_seed in values.items():
        for seed, v in zip(PROTO["seeds"], per_seed, strict=True):
            for n in (25, 100):
                for dist in ("id", "ood"):
                    rows.append(
                        {"method": method, "seed": seed, "real_label_budget": n, "distribution": dist, "relative_l2": v}
                    )
    return rows


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "PKG", tmp_path)
    shuffle = tmp_path / "shuffle.json"
    shuffle.write_text('{"mechanism_supported": true}')
    run_root = tmp_path / "run"
    run_root.mkdir()
    return run_root, shuffle


def _verdict(rows, workdir) -> str:
    run_root, shuffle = workdir
    out = gate.write_claim_artifacts(
        table_rows=rows,
        run_root=run_root,
        proto=PROTO,
        shuffle_path=shuffle,
        git_sha=None,
        out_stem="test_gate",
        write_paper_stub=False,
    )
    return out["verdict"]


def test_identical_seeds_are_not_testable(workdir):
    rows = _rows({"scratch": [2.1, 2.6, 1.5], "mml_direct": [0.7778] * 3, "lmop_jepa": [0.6857] * 3})
    assert _verdict(rows, workdir) == "NOT_TESTABLE_NO_SEED_VARIANCE"


def test_real_seed_variance_can_support(workdir):
    rows = _rows({"scratch": [0.6, 0.61, 0.62], "mml_direct": [0.50, 0.51, 0.505], "lmop_jepa": [0.40, 0.41, 0.405]})
    assert _verdict(rows, workdir) == "SUPPORTS_HYPOTHESIS"


def test_mml_better_falsifies(workdir):
    rows = _rows({"scratch": [0.6, 0.61, 0.62], "mml_direct": [0.40, 0.41, 0.405], "lmop_jepa": [0.50, 0.51, 0.505]})
    assert _verdict(rows, workdir) == "FALSIFIES_HYPOTHESIS"


def test_outputs_use_stem_not_canonical_name(workdir):
    rows = _rows({"scratch": [0.6, 0.61, 0.62], "mml_direct": [0.40, 0.41, 0.405], "lmop_jepa": [0.50, 0.51, 0.505]})
    _verdict(rows, workdir)
    tables = gate.PKG / "results" / "tables"
    assert (tables / "test_gate.json").exists()
    assert not (tables / "claim_gate.json").exists()
    assert not (gate.PKG / "paper" / "RESULTS_AUTO.md").exists()


def _support_rows():
    return _rows({"scratch": [0.6, 0.61, 0.62], "mml_direct": [0.50, 0.51, 0.505], "lmop_jepa": [0.40, 0.41, 0.405]})


def _report(workdir):
    return json.loads((workdir[0] / "claim_gate.json").read_text())


def test_wrong_seed_set_cannot_substitute_for_frozen_matrix(workdir):
    rows = [{**row, "seed": row["seed"] + 1000} for row in _support_rows()]
    assert _verdict(rows, workdir) == "INVALID_MATRIX"
    report = _report(workdir)
    assert not report["matrix_complete"]
    assert report["matrix_validation"]["observed_unique_cells"] == 0
    assert report["support_cells"] == []


def test_duplicate_cannot_conceal_missing_scratch_cell(workdir):
    rows = _support_rows()
    missing = rows[0]
    rows[0] = rows[1].copy()
    assert _verdict(rows, workdir) == "INVALID_MATRIX"
    report = _report(workdir)
    assert report["observed_eval_rows"] == report["expected_eval_rows"] == 36
    assert report["matrix_validation"]["missing_cells"] == [
        {key: missing[key] for key in ("method", "seed", "real_label_budget", "distribution")}
    ]
    assert any("duplicate" in error for error in report["matrix_validation"]["errors"])


def test_missing_cell_returns_incomplete_without_claims_or_nan(workdir):
    rows = _support_rows()[1:]
    assert _verdict(rows, workdir) == "INCOMPLETE_MATRIX"
    report = _report(workdir)
    assert report["comparisons"] == []
    assert report["support_cells"] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("extra", ["duplicate", "unknown_method", "unknown_distribution"])
def test_additional_rows_cannot_be_silently_ignored(workdir, extra):
    rows = _support_rows()
    row = rows[0].copy()
    if extra == "unknown_method":
        row["method"] = "extra_baseline"
    elif extra == "unknown_distribution":
        row["distribution"] = "exploratory"
    rows.append(row)
    assert _verdict(rows, workdir) == "INVALID_MATRIX"


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf"), -0.1, True, "0.6", None])
def test_invalid_primary_metric_in_secondary_reference_blocks_claim(workdir, metric):
    # Scratch is secondary scientifically, but it is required by the frozen matrix.
    rows = _support_rows()
    rows[0]["relative_l2"] = metric
    assert _verdict(rows, workdir) == "INVALID_MATRIX"
    json.dumps(_report(workdir), allow_nan=False)


@pytest.mark.parametrize("field,value", [("seed", 11.9), ("seed", True), ("seed", "11"), ("real_label_budget", 25.9)])
def test_fractional_or_coerced_identity_cannot_enter_matrix(workdir, field, value):
    rows = _support_rows()
    rows[0][field] = value
    assert _verdict(rows, workdir) == "INVALID_MATRIX"


@pytest.mark.parametrize("payload", [{"mechanism_supported": "false"}, {"mechanism_supported": 1}, {"mechanism_supported": None}, [], None])
def test_mechanism_evidence_requires_boolean(workdir, payload):
    workdir[1].write_text(json.dumps(payload))
    assert _verdict(_support_rows(), workdir) == "INCONCLUSIVE_MECHANISM"
    report = _report(workdir)
    assert not report["shuffled_control"]["mechanism_supported"]
    assert report["shuffled_control"]["validation_error"]


def test_invalid_shuffled_json_does_not_support_claim(workdir):
    workdir[1].write_text("not json")
    assert _verdict(_support_rows(), workdir) == "INCONCLUSIVE_MECHANISM"


def test_false_boolean_mechanism_remains_inconclusive(workdir):
    workdir[1].write_text('{"mechanism_supported": false}')
    assert _verdict(_support_rows(), workdir) == "INCONCLUSIVE_MECHANISM"
    assert _report(workdir)["shuffled_control"]["validation_error"] is None


def test_valid_matrix_is_invariant_to_row_order(workdir):
    assert _verdict(list(reversed(_support_rows())), workdir) == "SUPPORTS_HYPOTHESIS"
    assert _report(workdir)["matrix_validation"]["observed_unique_cells"] == 36


@pytest.mark.parametrize("seeds,subsets", [([], [25]), ([11, 11], [25]), ([11], [25, 25]), ([True], [25]), ([11], [25.5])])
def test_invalid_protocol_axes_fail_before_publication(workdir, seeds, subsets):
    proto = {"seeds": seeds, "data_sizes": {"confirmatory": {"subsets": subsets}}}
    with pytest.raises(ValueError, match="Protocol"):
        gate.write_claim_artifacts(
            table_rows=_support_rows(), run_root=workdir[0], proto=proto,
            shuffle_path=workdir[1], git_sha=None, write_paper_stub=False,
        )
    assert not (workdir[0] / "claim_gate.json").exists()
