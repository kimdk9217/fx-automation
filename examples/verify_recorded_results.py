"""Recomputes the reported numbers from the saved result files."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]


def ranks(values):
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2
        i = j
    return result


def spearman(values):
    x, y = list(range(len(values))), ranks(values)
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx)**2 for a in x) * sum((b - my)**2 for b in y))
    return num / den if den else 0.0


def main():
    first = json.loads((ROOT / "results/eval1/result.json").read_text())
    groups = defaultdict(list)
    for row in first["records"]:
        a, b = row["distances"][row["source_word"]], row["distances"][row["target_word"]]
        assert len(a) == len(b) == 11
        for source, target in zip(a, b):
            assert len(source) == len(target) == 16
            groups[row["instruction_key"]].append(spearman([x-y for x, y in zip(source, target)]))
    assert len(groups) == 10 and all(len(v) == 33 for v in groups.values())
    pair_scores = [median(v) for v in groups.values()]
    assert sum(v > 0 for v in pair_scores) == 9
    assert round(median(pair_scores), 3) == 0.525

    second = json.loads((ROOT / "results/eval2/results.json").read_text())
    truths = {r["curve_id"]: r for r in json.loads((ROOT / "results/eval2/ground_truth.json").read_text())}
    maes, boundaries, exact = [], [], []
    for row in second["case_scores"]:
        truth = truths[row["curve_id"]]
        target = truth["curve_1001"]
        assert len(target) == 1001
        if not row["valid_plan"]:
            maes.append(1.0); boundaries.append(1.0); exact.append(False)
            continue
        pred = row["predicted_curve_1001"]
        assert len(pred) == len(target)
        maes.append(mean(abs(a-b) for a, b in zip(pred, target)))
        exact.append(row["predicted_topology"] == truth["topology"])
        p, t = row["predicted_change_points"], truth["change_points"]
        count = max(len(p), len(t))
        boundaries.append((sum(abs(a-b) for a, b in zip(p, t)) + abs(len(p)-len(t))) / count if count else 0.0)
    assert len(maes) == 90 and sum(exact) == 71
    assert round(mean(maes), 3) == 0.152
    assert round(mean(boundaries), 3) == 0.171
    fixed = mean(mean(abs(v-0.5) for v in t["curve_1001"]) for t in truths.values())
    oracle = mean(mean(abs(v-median(t["curve_1001"])) for v in t["curve_1001"]) for t in truths.values())
    assert round(fixed, 3) == 0.214 and round(oracle, 3) == 0.191
    print(json.dumps({"eval1_positive_pairs": 9, "eval1_median_rho": median(pair_scores),
                      "eval2_correct": sum(exact), "eval2_cases": len(maes),
                      "eval2_mae": mean(maes), "eval2_boundary_error": mean(boundaries),
                      "fixed_constant_mae": fixed, "oracle_constant_mae": oracle}, indent=2))


if __name__ == "__main__":
    main()
