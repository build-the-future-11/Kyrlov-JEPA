"""Claim-gate logic for LMOP-JEPA (degenerate seed CIs must never count as support)."""

from __future__ import annotations

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
