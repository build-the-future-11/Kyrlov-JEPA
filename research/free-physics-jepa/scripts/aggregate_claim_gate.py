#!/usr/bin/env python3
"""Aggregate confirmatory metrics into claim-gate artifacts for publication."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PKG = Path(__file__).resolve().parents[1]


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def load_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["real_label_budget"] = int(r["real_label_budget"])
        for k in (
            "relative_l2",
            "h1_error",
            "energy_error",
            "flux_error",
            "pde_residual",
            "bootstrap_l2_ci_low",
            "bootstrap_l2_ci_high",
        ):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def seed_means(rows: list[dict], *, method: str, n: int, dist: str) -> list[float]:
    return [
        float(r["relative_l2"])
        for r in rows
        if r["method"] == method and int(r["real_label_budget"]) == n and r["distribution"] == dist
    ]


def paired_seed_deltas(rows: list[dict], *, n: int, dist: str) -> dict[str, Any]:
    """Per-seed (LMOP - MML) on relative L2; negative means LMOP better."""
    by_seed: dict[int, dict[str, float]] = {}
    for r in rows:
        if int(r["real_label_budget"]) != n or r["distribution"] != dist:
            continue
        if r["method"] not in ("mml_direct", "lmop_jepa"):
            continue
        by_seed.setdefault(int(r["seed"]), {})[r["method"]] = float(r["relative_l2"])
    deltas = []
    seeds = []
    for seed, d in sorted(by_seed.items()):
        if "mml_direct" in d and "lmop_jepa" in d:
            deltas.append(d["lmop_jepa"] - d["mml_direct"])
            seeds.append(seed)
    return {
        "seeds": seeds,
        "deltas_lmop_minus_mml": deltas,
        "mean_delta": _mean(deltas),
        "std_delta": _std(deltas),
        "all_lmop_better": bool(deltas) and all(d < 0 for d in deltas),
        "all_mml_better_or_tie": bool(deltas) and all(d >= 0 for d in deltas),
    }


def aggregate_cell(rows: list[dict], *, method: str, n: int, dist: str) -> dict[str, Any]:
    vals = seed_means(rows, method=method, n=n, dist=dist)
    # Across-seed "CI" as mean ± 1.96 * sem (descriptive; n_seeds=3)
    sem = _std(vals) / math.sqrt(len(vals)) if vals else float("nan")
    return {
        "method": method,
        "n": n,
        "distribution": dist,
        "n_seeds": len(vals),
        "mean_relative_l2": _mean(vals),
        "std_relative_l2": _std(vals),
        "ci_low": _mean(vals) - 1.96 * sem if vals else float("nan"),
        "ci_high": _mean(vals) + 1.96 * sem if vals else float("nan"),
        "per_seed": vals,
    }


def cis_nonoverlap(a: dict, b: dict) -> bool:
    return a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]


def seed_variance_degenerate(a: dict, b: dict) -> bool:
    """True when seeds do not vary either model (e.g. zero-shot: one checkpoint per method).

    A zero-width across-seed CI makes any nonzero difference "non-overlapping", so the
    CI criterion carries no statistical information and must not count as support.
    """
    tol = 1e-9
    return (
        a["n_seeds"] > 1
        and b["n_seeds"] > 1
        and a["std_relative_l2"] <= tol * max(1.0, abs(a["mean_relative_l2"]))
        and b["std_relative_l2"] <= tol * max(1.0, abs(b["mean_relative_l2"]))
    )


def write_claim_artifacts(
    *,
    table_rows: list[dict],
    run_root: Path,
    proto: dict,
    shuffle_path: Path,
    git_sha: str | None,
    out_stem: str = "claim_gate",
    write_paper_stub: bool = True,
    amendments: list[str] | None = None,
    source_csv: str = "results/tables/confirmatory.csv",
) -> dict[str, Any]:
    seeds = list(proto["seeds"])
    subsets = list(proto["data_sizes"]["confirmatory"]["subsets"])
    comparisons = []
    support_cells = []
    for n in subsets:
        for dist in ("id", "ood"):
            mml = aggregate_cell(table_rows, method="mml_direct", n=n, dist=dist)
            lmop = aggregate_cell(table_rows, method="lmop_jepa", n=n, dist=dist)
            scratch = aggregate_cell(table_rows, method="scratch", n=n, dist=dist)
            paired = paired_seed_deltas(table_rows, n=n, dist=dist)
            degenerate = seed_variance_degenerate(lmop, mml)
            lmop_beats_mml = bool(
                mml["n_seeds"] == len(seeds)
                and lmop["n_seeds"] == len(seeds)
                and paired["all_lmop_better"]
                and cis_nonoverlap(lmop, mml)
                and not degenerate
            )
            comparisons.append(
                {
                    "n": n,
                    "distribution": dist,
                    "scratch": scratch,
                    "mml_direct": mml,
                    "lmop_jepa": lmop,
                    "paired": paired,
                    "lmop_beats_mml_nonoverlap_ci": lmop_beats_mml,
                    "mml_beats_or_ties_lmop": paired["all_mml_better_or_tie"],
                    "seed_variance_degenerate": degenerate,
                }
            )
            if lmop_beats_mml:
                support_cells.append(f"N={n}/{dist}")

    shuffle = None
    mechanism_ok = False
    if shuffle_path.exists():
        shuffle = json.loads(shuffle_path.read_text())
        mechanism_ok = bool(shuffle.get("mechanism_supported"))

    expected_rows = len(seeds) * len(subsets) * 3 * 2  # methods × id/ood
    complete = len(table_rows) >= expected_rows

    if not complete:
        verdict = "INCOMPLETE_MATRIX"
        summary = f"Only {len(table_rows)}/{expected_rows} eval rows; do not claim."
    elif comparisons and all(c["seed_variance_degenerate"] for c in comparisons):
        verdict = "NOT_TESTABLE_NO_SEED_VARIANCE"
        summary = (
            "MML and LMOP predictions do not vary across seeds (single checkpoint per method), "
            "so the across-seed CI criterion is degenerate; the frozen gate cannot be applied."
        )
    elif support_cells and mechanism_ok:
        verdict = "SUPPORTS_HYPOTHESIS"
        summary = (
            f"LMOP beats MML with non-overlapping seed CIs on {support_cells}; "
            "shuffled-physics control worse than correct."
        )
    elif support_cells and not mechanism_ok:
        verdict = "INCONCLUSIVE_MECHANISM"
        summary = (
            f"LMOP numerically beats MML on {support_cells}, but shuffled control "
            "does not support physics-dependent mechanism."
        )
    elif any(c["mml_beats_or_ties_lmop"] for c in comparisons):
        verdict = "FALSIFIES_HYPOTHESIS"
        summary = "MML ≥ LMOP on relative L2 across seeds under this protocol."
    else:
        verdict = "INCONCLUSIVE"
        summary = "No cell meets non-overlapping CI support; differences mixed or overlapping."

    report = {
        "program": "lmop-jepa",
        "evidence_class": "CONFIRMATORY",
        "verdict": verdict,
        "summary": summary,
        "git_sha": git_sha,
        "run_root": str(run_root),
        "expected_eval_rows": expected_rows,
        "observed_eval_rows": len(table_rows),
        "matrix_complete": complete,
        "shuffled_control": {
            "path": str(shuffle_path) if shuffle_path.exists() else None,
            "mechanism_supported": mechanism_ok,
            "correct_final_loss": None if shuffle is None else shuffle.get("correct_final_loss"),
            "shuffled_final_loss": None if shuffle is None else shuffle.get("shuffled_final_loss"),
        },
        "comparisons": comparisons,
        "support_cells": support_cells,
        "language_rules": [
            "Do not claim novelty of manufactured solutions, FNO, or JEPA.",
            "State comparative transfer hypothesis only.",
            "Smoke metrics are not evidence.",
            "Krylov-JEPA is unrelated and must not be cited as LMOP support.",
        ],
        "amendments": amendments or list(proto.get("amendments", [])),
    }

    out_json = PKG / "results" / "tables" / f"{out_stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    (run_root / "claim_gate.json").write_text(json.dumps(report, indent=2))

    # Human-readable markdown for the paper results section
    lines = [
        "# Claim gate (auto-generated)",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        summary,
        "",
        f"- Matrix complete: {complete} ({len(table_rows)}/{expected_rows} rows)",
        f"- Shuffled mechanism supported: {mechanism_ok}",
        f"- Git SHA: `{git_sha or 'NONE'}`",
        f"- Run: `{run_root}`",
        "",
        "## Seed-aggregated relative L2",
        "",
        "| N | Dist | Scratch | MML | LMOP | Δ(LMOP−MML) | LMOP wins (CI) | Seed CI degenerate |",
        "|---|------|---------|-----|------|-------------|----------------|--------------------|",
    ]
    for c in comparisons:
        lines.append(
            "| {n} | {dist} | {s:.4f}±{ss:.4f} | {m:.4f}±{ms:.4f} | {l:.4f}±{ls:.4f} | {d:.4f} | {w} | {g} |".format(
                n=c["n"],
                dist=c["distribution"],
                s=c["scratch"]["mean_relative_l2"],
                ss=c["scratch"]["std_relative_l2"],
                m=c["mml_direct"]["mean_relative_l2"],
                ms=c["mml_direct"]["std_relative_l2"],
                l=c["lmop_jepa"]["mean_relative_l2"],
                ls=c["lmop_jepa"]["std_relative_l2"],
                d=c["paired"]["mean_delta"],
                w="yes" if c["lmop_beats_mml_nonoverlap_ci"] else "no",
                g="yes" if c["seed_variance_degenerate"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Publication language",
            "",
            f"Under this frozen protocol, evidence **{verdict.replace('_', ' ').lower()}** "
            "the hypothesis that LMOP-JEPA transfers better than MML-direct at small genuine budgets.",
            "",
        ]
    )
    md_path = PKG / "results" / "tables" / f"{out_stem}.md"
    md_path.write_text("\n".join(lines) + "\n")
    (run_root / "claim_gate.md").write_text("\n".join(lines) + "\n")

    if write_paper_stub:
        paper = PKG / "paper" / "RESULTS_AUTO.md"
        paper.parent.mkdir(parents=True, exist_ok=True)
        paper.write_text(
            "\n".join(
                [
                    "# Auto-filled results (do not edit by hand — regenerated by overnight run)",
                    "",
                    *lines[1:],
                    "",
                    f"Source CSV: `{source_csv}`",
                    f"Source gate: `results/tables/{out_stem}.json`",
                    "",
                ]
            )
        )

    report["artifact"] = str(out_json)
    return report


def main() -> int:
    import argparse

    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(PKG / "results" / "tables" / "confirmatory.csv"))
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--out-stem", type=str, default="claim_gate")
    p.add_argument("--no-paper-stub", action="store_true")
    p.add_argument("--shuffle-json", type=str, default=None)
    p.add_argument("--git-sha", type=str, default=None)
    args = p.parse_args()
    proto = yaml.safe_load((PKG / "protocol.yaml").read_text())
    rows = load_rows_from_csv(Path(args.csv))
    run_root = Path(args.run_root)
    sha = args.git_sha
    if sha is None and (run_root / "summary.json").exists():
        sha = json.loads((run_root / "summary.json").read_text()).get("git_sha")
    shuffle_path = Path(args.shuffle_json) if args.shuffle_json else run_root / "shuffled_control.json"
    try:
        source_csv = str(Path(args.csv).resolve().relative_to(PKG))
    except ValueError:
        source_csv = args.csv
    out = write_claim_artifacts(
        table_rows=rows,
        run_root=run_root,
        proto=proto,
        shuffle_path=shuffle_path,
        git_sha=sha,
        out_stem=args.out_stem,
        write_paper_stub=not args.no_paper_stub,
        source_csv=source_csv,
    )
    print(json.dumps({"verdict": out["verdict"], "artifact": out["artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
