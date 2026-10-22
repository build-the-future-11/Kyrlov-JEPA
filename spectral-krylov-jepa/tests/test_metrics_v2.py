"""Independent physical oracles and strict-cohort regressions for the v2 opt-in API."""

import base64
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spectral_krylov_jepa.evaluation import metrics as retained
from spectral_krylov_jepa.evaluation import metrics_v2 as strict
from spectral_krylov_jepa.physics.grid import GridSpec


def analytic_box(n=4, length=1.0):
    grid = GridSpec(n_interior=n, x_max=length, y_max=length)
    sine = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
    product = np.outer(sine, sine)
    psi = product / np.linalg.norm(product) / math.sqrt(grid.hx * grid.hy)
    energy = (1 - math.cos(math.pi / (n + 1))) * (1 / grid.hx**2 + 1 / grid.hy**2)
    return grid, np.zeros((n, n)), psi, energy


def example(case_id="analytic", prediction=None):
    grid, potential, psi, energy = analytic_box()
    return grid, {
        "case_id": case_id, "potential": potential.tolist(), "psi_true": psi.tolist(),
        "e_true": energy, "psi_hat": (psi if prediction is None else prediction).tolist(),
        "e_hat": energy,
    }


def test_zero_prediction_is_rejected_before_any_metric_record():
    grid, potential, psi, energy = analytic_box()
    zero = np.zeros_like(psi)
    with pytest.raises(ValueError, match="psi_hat is a zero state"):
        strict.evaluate_example(potential, psi, energy, zero, energy, grid)


@pytest.mark.parametrize("sign", [1, -1])
def test_analytic_ground_state_and_three_distinct_energy_choices(sign):
    grid, potential, psi, energy = analytic_box()
    metrics = strict.evaluate_example(potential, psi, energy, sign * psi, energy + 3, grid)
    assert metrics["fidelity"] == pytest.approx(1, abs=1e-14)
    assert metrics["infidelity"] == pytest.approx(0, abs=1e-14)
    assert metrics["sign_aligned_rel_l2"] == pytest.approx(0, abs=1e-14)
    assert metrics["rayleigh_energy"] == pytest.approx(energy, abs=2e-14)
    assert metrics["residual_rel"] == pytest.approx(3, abs=2e-14)
    assert metrics["residual_true_e"] < 2e-14
    assert metrics["residual_rayleigh"] < 2e-14
    assert metrics["rel_energy_error"] == pytest.approx(3 / (abs(energy) + 1e-6))


def test_non_eigenvector_matches_independent_dense_stencil():
    grid, potential, psi, energy = analytic_box(n=3)
    potential = np.arange(9, dtype=float).reshape(3, 3) / 7
    candidate = np.arange(1, 10, dtype=float).reshape(3, 3)
    candidate /= np.linalg.norm(candidate) * math.sqrt(grid.hx * grid.hy)
    # Construct the 5-point operator independently, without production builders.
    operator = np.zeros((9, 9))
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            operator[index, index] = 1 / grid.hx**2 + 1 / grid.hy**2 + potential[row, col]
            for dr, dc, spacing in ((0, -1, grid.hx), (0, 1, grid.hx),
                                    (-1, 0, grid.hy), (1, 0, grid.hy)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    operator[index, nr * 3 + nc] = -0.5 / spacing**2
    q = candidate.ravel() / np.linalg.norm(candidate)
    reference = psi.ravel() / np.linalg.norm(psi)
    hq = operator @ q
    rayleigh = q @ hq
    metrics = strict.evaluate_example(potential, psi, energy, candidate, 7.0, grid)
    assert metrics["rayleigh_energy"] == pytest.approx(rayleigh, rel=1e-14)
    assert metrics["fidelity"] == pytest.approx((q @ reference)**2, rel=1e-14)
    for key, e in (("residual_rel", 7.0), ("residual_true_e", energy),
                   ("residual_rayleigh", rayleigh)):
        assert metrics[key] == pytest.approx(np.linalg.norm(hq - e * q), rel=1e-14)
    assert metrics["sign_aligned_rel_l2"] == pytest.approx(np.linalg.norm(q - reference))


def test_large_domain_does_not_suppress_residual_with_norm_floor():
    grid, potential, psi, energy = analytic_box(length=1e150)
    actual = strict.evaluate_example(potential, psi, energy, psi, 1.0, grid)
    assert actual["residual_rel"] == pytest.approx(1.0, rel=1e-14)
    assert actual["rayleigh_energy"] == pytest.approx(energy, rel=1e-14, abs=0)
    assert actual["fidelity"] == pytest.approx(1.0)


@pytest.mark.parametrize("field,bad", [
    ("psi_hat", np.zeros((4, 4))), ("psi_true", np.zeros((4, 4))),
    ("psi_hat", np.full((4, 4), np.nan)), ("psi_true", np.full((4, 4), np.inf)),
    ("potential", np.full((4, 4), np.inf)), ("potential", np.zeros((4, 4), complex)),
    ("psi_hat", np.zeros((4, 3))), ("psi_hat", np.ones((4, 4), bool)),
    ("e_hat", float("nan")), ("e_true", float("inf")), ("e_hat", True),
    ("e_hat", 2**53 + 1), ("potential", np.full((4, 4), 2**53 + 1, dtype=np.int64)),
    ("potential", [[True, 0., 0., 0.]] + [[0.] * 4] * 3),
    ("potential", [[2**53 + 1, 0., 0., 0.]] + [[0.] * 4] * 3),
])
def test_invalid_inputs_are_explicit(field, bad):
    grid, case = example()
    case.pop("case_id")
    case[field] = bad
    with pytest.raises(ValueError):
        strict.evaluate_example(**case, grid=grid)


@pytest.mark.parametrize("scale", [0.5, 2.0, 1.0 + 2e-6])
def test_material_normalization_error_is_not_repaired(scale):
    grid, potential, psi, energy = analytic_box()
    with pytest.raises(ValueError, match="physical L2 norm"):
        strict.evaluate_example(potential, psi, energy, psi * scale, energy, grid)


def test_dependency_fidelity_assumption_requires_explicit_state_admission():
    grid, potential, psi, energy = analytic_box()
    old = retained.evaluate_example(potential, psi, energy, 2 * psi, energy, grid)
    assert old["fidelity"] == pytest.approx(4)
    assert old["infidelity"] == pytest.approx(-3)
    with pytest.raises(ValueError, match="physical L2 norm"):
        strict.evaluate_example(potential, psi, energy, 2 * psi, energy, grid)


def test_declared_normalization_roundoff_is_admitted():
    grid, potential, psi, energy = analytic_box()
    actual = strict.evaluate_example(potential, psi, energy, psi * (1 + 1e-8), energy, grid)
    assert actual["fidelity"] == pytest.approx(1)
    assert actual["residual_rel"] < 2e-14


@pytest.mark.parametrize("grid", [
    GridSpec(n_interior=4.0), GridSpec(n_interior=4, x_max=float("inf")),
    GridSpec(n_interior=4, x_min=float("nan")),
    GridSpec(n_interior=4, x_max=1e200, y_max=1e200),
    GridSpec(n_interior=4, x_max=1e-200, y_max=1e-200),
])
def test_unrepresentable_grids_are_rejected(grid):
    _, case = example()
    case.pop("case_id")
    with pytest.raises(ValueError):
        strict.evaluate_example(**case, grid=grid)


def test_cohort_retains_invalid_case_and_suppresses_aggregate():
    grid, valid = example()
    _, invalid = example("zero", np.zeros((4, 4)))
    report = strict.audit_examples([valid, invalid], grid)
    assert report["status"] == "INVALID"
    assert (report["n_requested"], report["n_valid"], report["n_invalid"]) == (2, 1, 1)
    assert report["summary"] is None
    assert report["cases"][0]["status"] == "VALID"
    assert report["cases"][1]["error"] == "psi_hat is a zero state"
    json.dumps(report, allow_nan=False)


def test_valid_cohort_uses_same_complete_denominator():
    grid, first = example()
    _, second = example("energy-error")
    second["e_hat"] += 4
    report = strict.audit_examples([first, second], grid)
    assert report["status"] == "VALID"
    assert report["summary"]["n"] == 2
    assert report["summary"]["residual_rel_mean"] == pytest.approx(2)
    assert report["summary"]["residual_rel_std"] == pytest.approx(2)
    assert all(len(report["summary"][key + "_values"]) == 2 for key in strict.METRIC_KEYS)


@pytest.mark.parametrize("cases", [[], [{}], [{"case_id": "missing"}]])
def test_missing_case_schema_rejected(cases):
    with pytest.raises(ValueError):
        strict.audit_examples(cases, GridSpec(n_interior=4))


def test_duplicate_ids_do_not_change_cohort_size():
    grid, case = example()
    with pytest.raises(ValueError, match="unique"):
        strict.audit_examples([case, copy.deepcopy(case)], grid)


@pytest.mark.parametrize("rows", [[], [{"fidelity": 1}],
    [dict.fromkeys(strict.METRIC_KEYS, float("nan"))],
    [dict.fromkeys(strict.METRIC_KEYS, 1), dict.fromkeys(strict.METRIC_KEYS, float("inf"))],
])
def test_summary_never_omits_nonfinite_or_incomplete_rows(rows):
    with pytest.raises(ValueError):
        strict.summarize_metrics(rows)


def test_dependency_nan_summary_changes_metric_denominator():
    complete = dict.fromkeys(strict.METRIC_KEYS, 1.0)
    incomplete = {**complete, "residual_rel": float("nan")}
    old = retained.summarize_metrics([complete, incomplete])
    assert old["n"] == 2
    assert old["residual_rel_mean"] == 1.0
    assert math.isnan(old["residual_rel_values"][1])
    with pytest.raises(ValueError, match="finite"):
        strict.summarize_metrics([complete, incomplete])


def test_summary_large_values_and_signed_cancellation():
    rows = [dict.fromkeys(strict.METRIC_KEYS, 1e308), dict.fromkeys(strict.METRIC_KEYS, -1e308)]
    summary = strict.summarize_metrics(rows)
    assert summary["rayleigh_energy_mean"] == 0
    assert summary["rayleigh_energy_std"] == pytest.approx(1e308)
    same = strict.summarize_metrics([rows[0], rows[0]])
    assert same["rayleigh_energy_mean"] == 1e308
    assert same["rayleigh_energy_std"] == 0


def test_nonzero_aggregate_underflow_is_explicit():
    rows = [dict.fromkeys(strict.METRIC_KEYS, 5e-324), dict.fromkeys(strict.METRIC_KEYS, 0)]
    with pytest.raises(ValueError, match="mean is not representable"):
        strict.summarize_metrics(rows)


@pytest.mark.parametrize("first", [1.0, 1e308])
def test_standard_deviation_centers_before_mean_rounding(first):
    second = float(np.nextafter(first, np.inf))
    rows = [dict.fromkeys(strict.METRIC_KEYS, first), dict.fromkeys(strict.METRIC_KEYS, second)]
    summary = strict.summarize_metrics(rows)
    assert summary["rayleigh_energy_std"] == pytest.approx((second - first) / 2, rel=1e-15)


def test_unrepresentable_nonzero_standard_deviation_is_refused():
    rows = [dict.fromkeys(strict.METRIC_KEYS, 5e-324), dict.fromkeys(strict.METRIC_KEYS, 1e-323)]
    with pytest.raises(ValueError, match="standard deviation"):
        strict.summarize_metrics(rows)


def run_cli(input_path, output_path):
    source = Path(strict.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(source), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, "-m", "spectral_krylov_jepa.evaluation.metrics_v2",
         str(input_path), str(output_path)], env=env, capture_output=True, text=True,
        timeout=30, check=False,
    )


@pytest.mark.parametrize("include_invalid", [False, True])
def test_actual_cli_retains_raw_input_source_identity_and_invalid_attempt(tmp_path, include_invalid):
    grid, case = example()
    cases = [case]
    if include_invalid:
        cases.append(example("invalid", np.zeros((4, 4)))[1])
    payload = {"evaluation_mode": "development", "grid": grid.to_dict(), "cases": cases}
    raw = json.dumps(payload, indent=1).encode()
    input_path, output_path = tmp_path / "input.json", tmp_path / "receipt.json"
    input_path.write_bytes(raw)
    run = run_cli(input_path, output_path)
    assert run.returncode == (2 if include_invalid else 0), run.stderr
    receipt = json.loads(output_path.read_text(), parse_constant=lambda x: pytest.fail(x))
    assert base64.b64decode(receipt["input_base64"]) == raw
    assert receipt["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["source_sha256"]["evaluation/metrics_v2.py"] == hashlib.sha256(
        Path(strict.__file__).read_bytes()).hexdigest()
    assert receipt["source_sha256"]["evaluation/metrics.py"] == hashlib.sha256(
        Path(retained.__file__).read_bytes()).hexdigest()
    assert receipt["environment"]["numpy"] == np.__version__
    if include_invalid:
        assert receipt["summary"] is None
        assert receipt["n_invalid"] == 1
    else:
        assert receipt["summary"]["n"] == 1
    before = output_path.read_bytes()
    again = run_cli(input_path, output_path)
    assert again.returncode == 2 and "cannot reserve receipt" in again.stderr
    assert output_path.read_bytes() == before


@pytest.mark.parametrize("raw", [
    b'{"evaluation_mode":"protected", "grid":{}, "cases":[]}',
    b'{"evaluation_mode":"development", "grid":{}, "cases":NaN}',
    b'{"evaluation_mode":"development", "evaluation_mode":"protected"}',
    b'not json', b'\xff',
])
def test_invalid_json_or_mode_still_leaves_original_bytes_in_cli_receipt(tmp_path, raw):
    input_path, output_path = tmp_path / "input.json", tmp_path / "receipt.json"
    input_path.write_bytes(raw)
    run = run_cli(input_path, output_path)
    assert run.returncode == 2
    receipt = json.loads(output_path.read_text())
    assert receipt["status"] == "INVALID"
    assert receipt["error"]
    assert receipt["summary"] is None
    assert base64.b64decode(receipt["input_base64"]) == raw
