"""Synthetic exporter regressions; no archived study inputs are accessed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


EXPORTER = Path(__file__).with_name("export_final_results.py")
SPEC = importlib.util.spec_from_file_location("export_final_results", EXPORTER)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "synthetic-study"
    for phase in ("initial", "reference_sensitivity"):
        folder = root / "results"
        if phase != "initial":
            folder /= phase
        folder.mkdir(parents=True, exist_ok=True)
        comparisons = []
        observations = {"quantum": [], "membrane": []}
        for index in range(32):
            domain = "quantum" if phase == "initial" and index < 16 else "membrane"
            family = f"synthetic-{index}"
            comparisons.append({
                "domain": domain, "n": 8, "family": family,
                "numerator": "candidate", "denominator": "reference",
                "primary": False, "ratio": 0.5, "coverage": 0.95,
                "ci": [0.4, 0.6], "draws": 100,
            })
            for seed in (11, 23):
                for method, seconds in (("candidate", 1.0), ("reference", 2.0)):
                    observations[domain].append({
                        "n": 8, "family": family, "method": method,
                        "potential_seed": seed, "training_seed": 7,
                        "capped_seconds": seconds,
                    })
        (folder / "comparisons.json").write_text(json.dumps(comparisons))
        for domain, rows in observations.items():
            if rows:
                filename = f"{domain}_test.csv" if phase == "initial" else "test.csv"
                pd.DataFrame(rows).to_csv(folder / filename, index=False)
    return root


def change_comparisons(root, phase, change):
    folder = root / "results"
    if phase != "initial":
        folder /= phase
    path = folder / "comparisons.json"
    path.write_text(json.dumps(change(json.loads(path.read_text()))))


def run_export(root, output):
    return subprocess.run(
        [sys.executable, str(EXPORTER), "--root", str(root), "--output", str(output)],
        text=True, capture_output=True, check=False,
    )


def result_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in (root / "results").rglob("*") if p.is_file()}


def test_complete_synthetic_table(study, tmp_path):
    before = result_bytes(study)
    output = tmp_path / "derived"
    result = run_export(study, output)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["comparisons"] == 64
    table = pd.read_csv(output / "final_results.csv")
    assert len(table) == 64
    assert set(table.phase) == {"initial", "reference_sensitivity"}
    assert set(table.domain) == {"quantum", "membrane"}
    assert (table.time_ratio == 0.5).all()
    assert result_bytes(study) == before


@pytest.mark.parametrize("kind", ["empty", "partial", "duplicate", "empty_phase"])
def test_incomplete_or_repeated_comparisons_rejected(study, tmp_path, kind):
    if kind == "empty":
        for phase in ("initial", "reference_sensitivity"):
            change_comparisons(study, phase, lambda rows: [])
    elif kind == "partial":
        change_comparisons(study, "initial", lambda rows: rows[:-1])
    elif kind == "duplicate":
        change_comparisons(study, "initial", lambda rows: rows[:-1] + [rows[0]])
    else:
        change_comparisons(study, "initial", lambda rows: rows + [
            {**row, "family": "extra-" + row["family"]} for row in rows
        ])
        for path in (study / "results").glob("*_test.csv"):
            data = pd.read_csv(path)
            extra = data.copy()
            extra["family"] = "extra-" + extra["family"]
            pd.concat([data, extra]).to_csv(path, index=False)
        change_comparisons(study, "reference_sensitivity", lambda rows: [])
    before = result_bytes(study)
    output = tmp_path / "rejected"
    result = run_export(study, output)
    assert result.returncode != 0
    assert '"status": "PASS"' not in result.stdout
    assert not output.exists()
    assert result_bytes(study) == before


def test_small_absolute_but_large_relative_ratio_error_rejected(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data.loc[(data.family == "synthetic-0") & (data.method == "candidate"), "capped_seconds"] = 2e-10
    data.to_csv(path, index=False)
    change_comparisons(study, "initial", lambda rows: [{**rows[0], "ratio": 2e-10}] + rows[1:])
    with pytest.raises(ValueError, match="ratio"):
        exporter.build(study)


@pytest.mark.parametrize("seconds", [-1.0, float("inf"), float("nan")])
def test_invalid_timings_rejected(study, seconds):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data.loc[data.family == "synthetic-0", "capped_seconds"] = seconds
    data.to_csv(path, index=False)
    change_comparisons(study, "initial", lambda rows: [{**rows[0], "ratio": 1.0}] + rows[1:])
    with pytest.raises(ValueError):
        exporter.build(study)


def test_matching_small_ratio_is_accepted(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data.loc[(data.family == "synthetic-0") & (data.method == "candidate"), "capped_seconds"] = 2e-10
    data.to_csv(path, index=False)
    change_comparisons(study, "initial", lambda rows: [{**rows[0], "ratio": 1e-10}] + rows[1:])
    assert len(exporter.build(study)) == 64


@pytest.mark.parametrize("kind", ["aggregate", "milliseconds"])
def test_overflow_cannot_produce_success(study, kind):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    if kind == "aggregate":
        data.loc[(data.family == "synthetic-0") & (data.method == "reference"), "capped_seconds"] = 1e308
        ratio = 0.0
    else:
        data.loc[data.family == "synthetic-0", "capped_seconds"] = 1e306
        ratio = 1.0
    data.to_csv(path, index=False)
    change_comparisons(study, "initial", lambda rows: [{**rows[0], "ratio": ratio}] + rows[1:])
    with pytest.raises(ValueError, match="finite"):
        exporter.build(study)


@pytest.mark.parametrize("change", [
    {"ci": [float("nan"), float("inf")]},
    {"ci": [0.6, 0.4]},
    {"coverage": "bad"},
    {"draws": 0},
    {"primary": "false"},
])
def test_invalid_interval_metadata_rejected(study, change):
    change_comparisons(study, "initial", lambda rows: [{**rows[0], **change}] + rows[1:])
    with pytest.raises(ValueError, match="metadata"):
        exporter.build(study)


def test_missing_pairing_keys_rejected(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data.loc[(data.family == "synthetic-0") & (data.potential_seed == 11), "potential_seed"] = float("nan")
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Pairing keys"):
        exporter.build(study)


def test_boolean_timings_rejected(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data["capped_seconds"] = True
    data.to_csv(path, index=False)
    change_comparisons(study, "initial", lambda rows: [{**row, "ratio": 1.0} for row in rows])
    with pytest.raises(ValueError, match="real numeric"):
        exporter.build(study)


def test_sample_count_uses_only_the_paired_methods(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    extra = data.iloc[[0]].copy()
    extra["method"] = "unrelated-method"
    extra["potential_seed"] = 99
    pd.concat([data, extra]).to_csv(path, index=False)
    table = exporter.build(study)
    row = table[(table.phase == "initial") & (table.family == "synthetic-0")].iloc[0]
    assert row.independent_physical_cases == 2


def test_unsigned_timings_preserve_negative_difference(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data["capped_seconds"] = np.where(
        data.method == "candidate", np.uint64(1 << 63), np.uint64(3 << 62),
    )
    data.to_csv(path, index=False)
    assert pd.read_csv(path).capped_seconds.dtype.kind == "u"
    change_comparisons(study, "initial", lambda rows: [
        {**row, "ratio": 2 / 3} if row["domain"] == "quantum" else row for row in rows
    ])
    table = exporter.build(study)
    quantum = table[table.domain == "quantum"]
    assert (quantum.absolute_difference_ms == -1000 * (1 << 62)).all()


@pytest.mark.parametrize("location", ["direct", "nested", "symlink"])
def test_output_cannot_write_into_results(study, tmp_path, location):
    if location == "direct":
        output = study / "results" / "derived"
    elif location == "nested":
        output = study / "results" / "reference_sensitivity" / "nested" / "derived"
    else:
        alias = tmp_path / "results-alias"
        alias.symlink_to(study / "results", target_is_directory=True)
        output = alias / "derived"
    before = result_bytes(study)
    result = run_export(study, output)
    assert result.returncode != 0
    assert not output.exists()
    assert result_bytes(study) == before


@pytest.mark.parametrize("route", ["through_results", "resolved_target"])
def test_outward_phase_symlink_remains_protected(study, tmp_path, route):
    phase = study / "results" / "reference_sensitivity"
    retained = tmp_path / "retained-sensitivity"
    phase.rename(retained)
    phase.symlink_to(retained, target_is_directory=True)
    before = {p.name: p.read_bytes() for p in retained.iterdir()}
    output = (phase if route == "through_results" else retained) / "derived"
    result = run_export(study, output)
    assert result.returncode != 0
    assert not output.exists()
    assert {p.name: p.read_bytes() for p in retained.iterdir()} == before


def test_existing_output_is_not_overwritten(study, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "final_results.csv"
    marker.write_text("preserve me\n")
    result = run_export(study, output)
    assert result.returncode != 0
    assert marker.read_text() == "preserve me\n"


def test_unpaired_observations_rejected(study):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    data.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unpaired"):
        exporter.build(study)


@pytest.mark.parametrize("candidate,reference,dtype,expected_ms", [
    ([2**53, 2**53], [2**53 + 1, 2**53 + 1], "i", -1000.0),
    ([2**53 + 1, 2**53 + 1], [2**53, 2**53], "i", 1000.0),
    ([2**63, 2**63], [2**63 + 1, 2**63 + 1], "u", -1000.0),
    ([2**63 + 1, 2**63 + 1], [2**63, 2**63], "u", 1000.0),
    ([2**60, 1], [1, 2**60 - 2], "i", 1000.0),
])
def test_integer_paired_differences_survive_float_precision(
    study, candidate, reference, dtype, expected_ms,
):
    path = study / "results" / "quantum_test.csv"
    data = pd.read_csv(path)
    times = {
        (method, seed): values[index]
        for method, values in (("candidate", candidate), ("reference", reference))
        for index, seed in enumerate((11, 23))
    }
    data["capped_seconds"] = [
        times[(row.method, row.potential_seed)] for row in data.itertuples()
    ]
    data.to_csv(path, index=False)
    assert pd.read_csv(path).capped_seconds.dtype.kind == dtype
    ratio = sum(candidate) / sum(reference)
    change_comparisons(study, "initial", lambda rows: [
        {**row, "ratio": ratio} if row["domain"] == "quantum" else row
        for row in rows
    ])
    before = result_bytes(study)
    table = exporter.build(study)
    assert len(table) == 64
    assert (table[table.domain == "quantum"].absolute_difference_ms == expected_ms).all()
    assert result_bytes(study) == before


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_source_hash_and_comparisons_use_the_same_csv_snapshot(study, monkeypatch, change):
    path = study / "results" / "quantum_test.csv"
    original = path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    replacement = pd.read_csv(path)
    replacement["capped_seconds"] *= 2
    replacement_bytes = replacement.to_csv(index=False).encode()
    real_read_csv = pd.read_csv
    changed = False

    def read_then_change_source(*args, **kwargs):
        nonlocal changed
        frame = real_read_csv(*args, **kwargs)
        if not changed:
            changed = True
            if change == "replace":
                path.write_bytes(replacement_bytes)
            else:
                path.unlink()
        return frame

    monkeypatch.setattr(exporter.pd, "read_csv", read_then_change_source)
    table = exporter.build(study)
    quantum = table[table.domain == "quantum"]
    assert len(table) == 64
    assert len(quantum) == 16
    assert (quantum.source_sha256 == original_hash).all()
    assert (quantum.numerator_mean_ms == 1000).all()
    assert (quantum.baseline_mean_ms == 2000).all()
    assert (quantum.absolute_difference_ms == -1000).all()
