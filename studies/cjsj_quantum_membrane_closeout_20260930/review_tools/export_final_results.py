"""Export a canonical comparison table from preserved per-case measurements.

Never writes under results/. Outputs derived files to an explicit new directory.
Archived timing observations, not verification-replay timings, are claim bearing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COMPARISONS = 64


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def comparison_specs(root):
    """Check structural coverage without claiming private archive identity."""
    specs = []
    identities = set()
    for phase, folder in (
        ("initial", root / "results"),
        ("reference_sensitivity", root / "results/reference_sensitivity"),
    ):
        comparisons = json.loads((folder / "comparisons.json").read_text())
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError(f"Missing comparison list for phase {phase}")
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise ValueError("Comparison must be an object")
            domain = comparison.get("domain", "membrane")
            if domain not in ("quantum", "membrane"):
                raise ValueError("Unknown comparison domain")
            grid = comparison.get("n")
            if type(grid) is not int or grid <= 0:
                raise ValueError("Comparison grid must be a positive integer")
            labels = tuple(comparison.get(key) for key in ("family", "numerator", "denominator"))
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("Comparison identity labels must be nonempty strings")
            coverage = comparison.get("coverage")
            interval = comparison.get("ci")
            draws = comparison.get("draws")
            if (
                not finite_number(coverage) or not 0 < coverage < 1
                or not isinstance(interval, list) or len(interval) != 2
                or not all(finite_number(value) for value in interval)
                or interval[0] > interval[1]
                or type(draws) is not int or draws <= 0
                or type(comparison.get("primary")) is not bool
            ):
                raise ValueError("Invalid comparison interval metadata")
            identity = (phase, domain, grid, *labels)
            if identity in identities:
                raise ValueError("Duplicate comparison identity")
            identities.add(identity)
            specs.append((phase, folder, domain, comparison))
    if len(specs) != EXPECTED_COMPARISONS:
        raise ValueError(f"Expected {EXPECTED_COMPARISONS} unique comparisons, found {len(specs)}")
    return specs


def build(root):
    rows = []
    for phase, folder, domain, c in comparison_specs(root):
        f = folder / ("test.csv" if phase != "initial" else f"{domain}_test.csv")
        d = pd.read_csv(f)
        d = d[(d.n == c["n"]) & (d.family == c["family"])]
        a = d[d.method == c["numerator"]].copy()
        b = d[d.method == c["denominator"]].copy()
        keys = ["potential_seed"] + (["training_seed"] if domain == "quantum" else [])
        if any(frame[keys].isna().any().any() for frame in (a, b)):
            raise ValueError("Pairing keys must not contain missing values")
        a = a.set_index(keys).sort_index()
        b = b.set_index(keys).sort_index()
        if a.empty or not a.index.equals(b.index) or a.index.has_duplicates:
            raise ValueError("Unpaired, empty or duplicate observations")
        av = a.capped_seconds.to_numpy()
        bv = b.capped_seconds.to_numpy()
        if any(not (
            np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ) for values in (av, bv)):
            raise ValueError("Timing observations must be real numeric values")
        av = av.astype(np.float64, copy=False)
        bv = bv.astype(np.float64, copy=False)
        if any(not np.isfinite(values).all() or (values < 0).any() for values in (av, bv)):
            raise ValueError("Timing observations must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            numerator_mean = float(av.mean())
            baseline_mean = float(bv.mean())
            difference_mean = float((av - bv).mean())
        if not all(math.isfinite(value) for value in (numerator_mean, baseline_mean, difference_mean)):
            raise ValueError("Aggregate timing observations must be finite")
        if baseline_mean <= 0:
            raise ValueError("Baseline mean time must be positive")
        ratio = numerator_mean / baseline_mean
        stored_ratio = c["ratio"]
        if (
            not finite_number(stored_ratio)
            or not math.isfinite(ratio)
            or not np.isclose(ratio, stored_ratio, rtol=1e-12, atol=0)
        ):
            raise ValueError("Stored ratio disagrees with measurements")
        derived = {
            "numerator_mean_ms": 1000 * numerator_mean,
            "baseline_mean_ms": 1000 * baseline_mean,
            "absolute_difference_ms": 1000 * difference_mean,
            "relative_difference_pct": 100 * (ratio - 1),
        }
        if not all(math.isfinite(value) for value in derived.values()):
            raise ValueError("Derived comparison values must be finite")
        rows.append(dict(
            study="CJSJ-QM-20260930", phase=phase, domain=domain, grid=c["n"],
            family=c["family"], numerator=c["numerator"], baseline=c["denominator"],
            primary=c["primary"], independent_physical_cases=a.reset_index().potential_seed.nunique(),
            training_seeds=a.reset_index().training_seed.nunique() if domain == "quantum" else 0,
            numerator_mean_ms=derived["numerator_mean_ms"], baseline_mean_ms=derived["baseline_mean_ms"],
            absolute_difference_ms=derived["absolute_difference_ms"], time_ratio=ratio,
            relative_difference_pct=derived["relative_difference_pct"], ci_coverage=c["coverage"],
            ci_low=c["ci"][0], ci_high=c["ci"][1], bootstrap_draws=c["draws"],
            source=str(f.relative_to(root)), source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        ))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    output = a.output.resolve()
    lexical_output = Path(os.path.abspath(a.output))
    lexical_results = {Path(os.path.abspath(a.root)) / "results", root / "results"}
    consumed_directories = (
        (root / "results").resolve(),
        (root / "results/reference_sensitivity").resolve(),
    )
    if (
        any(lexical_output == path or path in lexical_output.parents for path in lexical_results)
        or any(output == path or path in output.parents for path in consumed_directories)
    ):
        p.error("Output must be outside the protected results directory")
    t = build(root)
    output.mkdir(parents=True, exist_ok=False)
    t.to_csv(output / "final_results.csv", index=False)
    (output / "README.md").write_text('# Canonical comparison table\n\n64 comparisons from the retained initial and reference-sensitivity studies. CIs are paired bootstrap intervals; grid resolutions reuse physical cases. Timing repetitions and model seeds do not multiply independent physical sample counts. Primary interval coverage differs by study. Secondary intervals are descriptive and not jointly adjusted.\n')
    print(json.dumps({"status": "PASS", "comparisons": len(t), "output": str(output / "final_results.csv")}))


if __name__ == "__main__":
    main()
