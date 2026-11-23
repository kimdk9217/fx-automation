"""Evaluation 2 (curve reconstruction): constant-curve baselines."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/review/roundtrip_constant_baselines_v1"
METHODS = ("planner", "fixed_midpoint", "oracle_constant")
TOLERANCE = 1e-12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mae(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.shape != target.shape:
        raise ValueError("MAE requires identical sample grids")
    return float(np.mean(np.abs(predicted - target)))


def constants(target: np.ndarray) -> dict[str, float]:
    value = float(np.median(target))
    return {
        "fixed_midpoint": mae(np.full_like(target, 0.5), target),
        "oracle_constant": mae(np.full_like(target, value), target),
        "oracle_value": value,
    }


def counts(differences: np.ndarray) -> dict[str, int]:
    return {
        "planner_lower_mae": int(np.sum(differences > TOLERANCE)),
        "tie": int(np.sum(np.abs(differences) <= TOLERANCE)),
        "planner_higher_mae": int(np.sum(differences < -TOLERANCE)),
    }


def self_test() -> None:
    result = constants(np.array([0.0, 0.0, 1.0]))
    assert result["fixed_midpoint"] == 0.5
    assert result["oracle_value"] == 0.0
    assert np.isclose(result["oracle_constant"], 1.0 / 3.0)
    assert constants(np.array([0.2, 0.2, 0.2]))["oracle_constant"] == 0.0
    rng = np.random.default_rng(19)
    for _ in range(20):
        target = rng.random(101)
        best = constants(target)
        assert best["oracle_constant"] <= best["fixed_midpoint"] + TOLERANCE
        candidates = np.linspace(0.0, 1.0, 201)
        candidate_errors = np.mean(np.abs(target[None, :] - candidates[:, None]), axis=1)
        assert best["oracle_constant"] <= candidate_errors.min() + TOLERANCE
    assert counts(np.array([0.1, 0.0, -0.1])) == {
        "planner_lower_mae": 1, "tie": 1, "planner_higher_mae": 1
    }


def run() -> None:
    self_test()
    protocol_path = OUT / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    source = ROOT / protocol["input_directory"]
    if (OUT / "RESULTS.json").exists() or (OUT / "REPORT.md").exists():
        raise FileExistsError("Analysis output exists; refusing to overwrite")
    for name, expected in protocol["input_sha256"].items():
        assert digest(source / name) == expected, name
    truth = json.loads((source / "GROUND_TRUTH.json").read_text())
    original = json.loads((source / "RESULTS.json").read_text())
    instructions = json.loads((source / "PLANNER_CASES.json").read_text())
    targets = {row["curve_id"]: row for row in truth}
    assert len(targets) == 30 and len(instructions) == 90
    scores = original["case_scores"]
    assert len(scores) == 90
    assert len({row["case_id"] for row in scores}) == 90
    assert {row["case_id"] for row in scores} == {row["case_id"] for row in instructions}
    family_counts = Counter(row["family"] for row in truth)
    assert len(family_counts) == 6 and set(family_counts.values()) == {5}
    by_curve = defaultdict(list)
    case_rows = []
    invalid_ids = []
    max_stored_difference = 0.0
    for case in sorted(scores, key=lambda item: item["case_id"]):
        target_record = targets[case["curve_id"]]
        assert case["family"] == target_record["family"]
        target = np.asarray(target_record["curve_1001"], dtype=float)
        assert target.shape == (1001,) and np.isfinite(target).all()
        assert target.min() >= -TOLERANCE and target.max() <= 1.0 + TOLERANCE
        baseline = constants(target)
        if case["valid_plan"]:
            predicted = np.asarray(case["predicted_curve_1001"], dtype=float)
            assert predicted.shape == target.shape and np.isfinite(predicted).all()
            assert predicted.min() >= -TOLERANCE and predicted.max() <= 1.0 + TOLERANCE
            score = mae(predicted, target)
        else:
            assert case["predicted_curve_1001"] is None
            score = 1.0
            invalid_ids.append(case["case_id"])
        difference = abs(score - case["metrics"]["curve_mae"])
        max_stored_difference = max(max_stored_difference, difference)
        assert difference < TOLERANCE
        row = {
            "case_id": case["case_id"], "curve_id": case["curve_id"],
            "family": case["family"], "valid_plan": bool(case["valid_plan"]),
            "planner": score, **baseline,
        }
        case_rows.append(row)
        by_curve[case["curve_id"]].append(row)
    assert len(invalid_ids) == 2
    curve_rows = []
    for curve_id in sorted(targets):
        rows = by_curve[curve_id]
        assert len(rows) == 3
        assert len({row["oracle_value"] for row in rows}) == 1
        curve_rows.append({
            "curve_id": curve_id, "family": targets[curve_id]["family"],
            "oracle_value": rows[0]["oracle_value"],
            "invalid_cases": sum(not row["valid_plan"] for row in rows),
            **{method: float(np.mean([row[method] for row in rows])) for method in METHODS},
        })
    means = {method: float(np.mean([row[method] for row in curve_rows])) for method in METHODS}
    for method in METHODS:
        assert np.isclose(means[method], np.mean([row[method] for row in case_rows]), atol=TOLERANCE)
    old_summary = original["aggregate_curve_level_stratified_bootstrap_10000"]
    assert abs(means["planner"] - old_summary["curve_mae"]["mean"]) < TOLERANCE

    repetitions = protocol["uncertainty"]["replicates"]
    rng = np.random.default_rng(protocol["uncertainty"]["seed"])
    family_indices = {
        family: [i for i, row in enumerate(curve_rows) if row["family"] == family]
        for family in sorted(family_counts)
    }
    draws = np.empty((repetitions, 30), dtype=int)
    for replicate in range(repetitions):
        selected = []
        for indices in family_indices.values():
            selected.extend(indices[int(index)] for index in rng.integers(0, len(indices), len(indices)))
        draws[replicate] = selected
    planner_values = np.array([row["planner"] for row in curve_rows])
    planner_interval = np.quantile(planner_values[draws].mean(1), [0.025, 0.975])
    assert np.allclose(planner_interval, [old_summary["curve_mae"]["bootstrap_95_lower"], old_summary["curve_mae"]["bootstrap_95_upper"]], atol=TOLERANCE)
    comparisons = {}
    for method in METHODS[1:]:
        differences = np.array([row[method] - row["planner"] for row in curve_rows])
        case_differences = np.array([row[method] - row["planner"] for row in case_rows])
        interval = np.quantile(differences[draws].mean(1), [0.025, 0.975])
        comparisons[method] = {
            "mean_mae_difference": float(differences.mean()),
            "paired_95pct_bootstrap_interval": interval.tolist(),
            "relative_mae_reduction": float(differences.mean() / means[method]),
            "curve_level_comparison": counts(differences),
            "case_level_comparison": counts(case_differences),
        }
    families = []
    for family, indices in family_indices.items():
        rows = [curve_rows[index] for index in indices]
        families.append({
            "family": family, "curves": len(rows), "cases": 3 * len(rows),
            "invalid_cases": sum(row["invalid_cases"] for row in rows),
            **{method: float(np.mean([row[method] for row in rows])) for method in METHODS},
        })
    for name, expected in protocol["input_sha256"].items():
        assert digest(source / name) == expected, name
    result = {
        "analysis": protocol["analysis"], "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path), "analysis_code_sha256": digest(Path(__file__)),
        "input_sha256": protocol["input_sha256"], "denominator": protocol["denominator"],
        "invalid_case_ids": invalid_ids, "max_recomputed_vs_saved_mae_difference": max_stored_difference,
        "aggregate_mae": means, "planner_95pct_bootstrap_interval": planner_interval.tolist(),
        "paired_comparisons": comparisons, "family_rows": families,
        "curve_rows": curve_rows, "case_rows": case_rows,
        "scope_limits": protocol["scope_limits"],
        "reporting_note": "Retrospective comparison. Invalid planner outputs retain the original maximum-error convention; comparator outputs are curves, not executable event plans.",
    }
    labels = {"planner": "Planner (invalid cases = 1)", "fixed_midpoint": "Fixed 0.5", "oracle_constant": "Oracle constant (target median)"}
    lines = [
        "# Constant-curve comparison", "",
        "This is a retrospective comparison using the frozen 90-case round trip, not a new model run.",
        "Thirty target curves, three descriptions each; 1001 normalized-time scoring points. Both invalid planner outputs remain in the primary average with MAE 1, as in the original evaluation.", "",
        "## Aggregate MAE", "", "| Method | Mean MAE |", "| --- | ---: |",
        *[f"| {labels[method]} | {means[method]:.6f} |" for method in METHODS], "",
        "Fixed 0.5 ignores both the instruction and the target. The oracle constant uses the median of each target curve, which minimizes MAE among constant curves. It is a privileged reference, not a deployable baseline.", "",
        "## Paired comparisons", "",
        "Differences are comparator MAE minus planner MAE; positive means lower planner error. Three descriptions stay clustered within each curve. Intervals use 10,000 paired resamples stratified by six curve families, with the original seed 20260827.", "",
    ]
    for method, comparison in comparisons.items():
        low, high = comparison["paired_95pct_bootstrap_interval"]
        lines.extend([
            f"### {labels[method]}", "",
            f"- Mean MAE difference: {comparison['mean_mae_difference']:+.6f}; 95% paired bootstrap interval [{low:+.6f}, {high:+.6f}].",
            f"- Aggregate MAE reduction: {100 * comparison['relative_mae_reduction']:.2f}%.",
            f"- Curve-level lower/tied/higher planner error: {comparison['curve_level_comparison']}.",
            f"- Case-level lower/tied/higher planner error: {comparison['case_level_comparison']}.", "",
        ])
    lines.extend(["## All curve families", "", "| Family | Cases | Invalid | Planner MAE | Fixed 0.5 MAE | Oracle MAE |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in families:
        lines.append(f"| {row['family']} | {row['cases']} | {row['invalid_cases']} | {row['planner']:.6f} | {row['fixed_midpoint']:.6f} | {row['oracle_constant']:.6f} |")
    lines.extend([
        "", "## Interpretation limits", "",
        "- This tests curve reconstruction against no-temporal-variation references. It does not establish superiority over another language-aware planner.",
        "- The oracle uses ground-truth information unavailable to the planner and must be reported separately.",
        "- No event-sequence or change-point score is assigned to constant curves; no event plans were fabricated for them.",
        "- Bootstrap intervals describe this fixed synthetic benchmark, not generalization to human-authored instructions.",
        "- No audio quality, musical suitability, or structure-localization claim follows from these scores.",
        "- All 90 case rows and 30 curve rows are saved in RESULTS.json. No cases were excluded or regenerated.",
        "- The manuscript and original experimental artifacts were not edited.", "",
    ])
    with (OUT / "RESULTS.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (OUT / "REPORT.md").open("x") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"aggregate_mae": means, "paired_comparisons": comparisons, "family_rows": families, "invalid_case_ids": invalid_ids}, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Self-tests passed; no experimental data loaded.")
    elif sys.argv[1:]:
        raise SystemExit("Usage: eval2_baselines.py [--self-test]")
    else:
        run()
