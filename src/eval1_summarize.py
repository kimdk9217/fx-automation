"""Evaluation 1 (style transitions): summary table, confidence intervals and figure."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "demo_outputs/transition10_dry_piano_v1_run1/result.json"
OUTPUT = ROOT / "demo_outputs/transition10_direction_only_v1"
RESULT = OUTPUT / "RESULTS.json"
SUMMARY = OUTPUT / "RESULTS_SUMMARY.md"
TABLE = ROOT / "tables/paper/transition10_direction_only_v1.csv"
FIGURE_PNG = ROOT / "figs/paper/fig_transition10_direction_only_v1.png"
FIGURE_PDF = ROOT / "figs/paper/fig_transition10_direction_only_v1.pdf"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_826
EXAMPLE_KEY = "T_heavy_to_harsh"
EXAMPLE_RUN = "T_heavy_to_harsh__run1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(values), len(values))
        estimates[index] = np.mean(values[selected])
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def collect_pair_rows(records: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["instruction_key"]].append(record)

    pair_rows = []
    for key, runs in grouped.items():
        source = runs[0]["source_word"]
        target = runs[0]["target_word"]
        track_plan_rhos = np.asarray([
            row["score"]
            for run in runs
            for row in run["scoring"]["per_track"]
            if row["score"] is not None
        ], dtype=float)
        margin_changes = []
        source_changes = []
        target_changes = []
        for run in runs:
            source_distance = np.asarray(run["distances"][source], dtype=float)
            target_distance = np.asarray(run["distances"][target], dtype=float)
            margin = source_distance - target_distance
            margin_changes.extend((margin[:, -1] - margin[:, 0]).tolist())
            source_changes.extend(
                (source_distance[:, -1] - source_distance[:, 0]).tolist()
            )
            target_changes.extend(
                (target_distance[:, -1] - target_distance[:, 0]).tolist()
            )
        pair_rows.append({
            "style_pair": f"{source} → {target}",
            "source_word": source,
            "target_word": target,
            "separability": float(runs[0]["separability"]),
            "trajectory_rho_median": float(np.median(track_plan_rhos)),
            "trajectory_rho_q25": float(np.quantile(track_plan_rhos, 0.25)),
            "trajectory_rho_q75": float(np.quantile(track_plan_rhos, 0.75)),
            "positive_track_plan_count": int(np.sum(track_plan_rhos > 0)),
            "track_plan_count": int(len(track_plan_rhos)),
            "relative_margin_change_median": float(np.median(margin_changes)),
            "source_distance_change_median": float(np.median(source_changes)),
            "target_distance_change_median": float(np.median(target_changes)),
            "target_direction": bool(np.median(track_plan_rhos) > 0),
        })
    return pair_rows, grouped


def write_table(rows: list[dict]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "style_pair",
            "trajectory_rho_median",
            "rho_q25",
            "rho_q75",
            "positive_track_plan_count",
            "track_plan_count",
            "relative_margin_change_median",
            "target_direction",
        ])
        for row in rows:
            writer.writerow([
                row["style_pair"],
                f'{row["trajectory_rho_median"]:.6f}',
                f'{row["trajectory_rho_q25"]:.6f}',
                f'{row["trajectory_rho_q75"]:.6f}',
                row["positive_track_plan_count"],
                row["track_plan_count"],
                f'{row["relative_margin_change_median"]:.6f}',
                "yes" if row["target_direction"] else "no",
            ])


def write_summary(result: dict) -> None:
    summary = result["summary"]
    lines = [
        "# Source-to-target trajectory evaluation",
        "",
        "This is a presentation summary of the frozen system renders, not a new run.",
        "The untreated-audio comparison is intentionally omitted.",
        "",
        f'- Style pairs: {summary["n_style_pairs"]}',
        f'- Plans: {summary["n_plans"]}',
        f'- Acoustic-piano clips: {summary["n_tracks"]}',
        f'- Time windows per clip: {summary["n_windows"]}',
        f'- Median pair trajectory rho: {summary["pair_rho_median"]:.3f}',
        f'- Mean pair trajectory rho: {summary["pair_rho_mean"]:.3f}',
        f'- Mean rho 95% pair-bootstrap CI: [{summary["pair_rho_mean_ci_95"][0]:.3f}, {summary["pair_rho_mean_ci_95"][1]:.3f}]',
        f'- Pairs moving in the requested relative direction: {summary["positive_pairs"]}/{summary["n_style_pairs"]}',
        "",
        "Metric: Spearman correlation between time-window order and",
        "distance-to-source minus distance-to-target. Positive values indicate",
        "relative movement from the source-style anchor toward the target-style anchor.",
        "",
        "| Style transition | Median rho | Middle 50% | Positive measurements | Relative direction |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in result["pair_rows"]:
        lines.append(
            f'| {row["style_pair"]} | {row["trajectory_rho_median"]:+.3f} | '
            f'[{row["trajectory_rho_q25"]:+.3f}, {row["trajectory_rho_q75"]:+.3f}] | '
            f'{row["positive_track_plan_count"]}/{row["track_plan_count"]} | '
            f'{"Yes" if row["target_direction"] else "No"} |'
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(rows: list[dict], grouped: dict[str, list[dict]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    blue = "#24557A"
    orange = "#B45F32"
    dark = "#30363A"
    gray = "#A5ADB2"

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [1.12, 1.45], "wspace": 0.38},
    )

    example_run = next(
        run for run in grouped[EXAMPLE_KEY] if run["run"] == EXAMPLE_RUN
    )
    source = example_run["source_word"]
    target = example_run["target_word"]
    for word, color, label in (
        (source, blue, f"distance to {source}"),
        (target, orange, f"distance to {target}"),
    ):
        values = np.asarray(example_run["distances"][word], dtype=float)
        center = np.median(values, axis=0)
        low, high = np.quantile(values, [0.25, 0.75], axis=0)
        x = np.arange(1, values.shape[1] + 1)
        left.plot(x, center, color=color, lw=1.7, label=label)
        left.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
    example_rho = float(np.median([
        row["score"] for row in example_run["scoring"]["per_track"]
        if row["score"] is not None
    ]))
    left.set_title(
        f"Example: {source} → {target}  (ρ={example_rho:.2f})",
        loc="left", fontsize=8.8,
    )
    left.set_xlabel("time window")
    left.set_ylabel("distance to style anchor")
    left.set_xlim(1, 16)
    left.legend(frameon=False, fontsize=7, loc="best")
    left.text(
        0.0, -0.18, "selected successful plan; median across 11 clips",
        transform=left.transAxes, ha="left", va="top", fontsize=6.6, color="#626B70",
    )

    ordered = sorted(rows, key=lambda row: row["trajectory_rho_median"])
    y = np.arange(len(ordered))
    dot_colors = [blue if row["target_direction"] else gray for row in ordered]
    dot_values = [row["trajectory_rho_median"] for row in ordered]
    right.scatter(
        dot_values, y, s=42, color=dot_colors,
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    for position, row in zip(y, ordered):
        value = row["trajectory_rho_median"]
        right.text(
            value + 0.025, position, f"{value:+.2f}",
            ha="left", va="center", fontsize=6.8, color=dark,
        )
    right.axvline(0, color=dark, lw=0.7, linestyle=":")
    right.set_yticks(y)
    right.set_yticklabels([row["style_pair"] for row in ordered])
    right.set_xlabel("trajectory correlation  ρ")
    right.set_title("All 10 transitions", loc="left", fontsize=8.8)
    right.set_xlim(-0.16, 0.73)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.text(
        1.0, -0.18,
        "dot: median over 3 plans × 11 clips",
        transform=right.transAxes, ha="right", va="top", fontsize=6.6, color="#626B70",
    )

    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG)
    fig.savefig(FIGURE_PDF)
    plt.close(fig)


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows, grouped = collect_pair_rows(source["records"])
    values = np.asarray([row["trajectory_rho_median"] for row in rows], dtype=float)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "run_version": "transition10-direction-only-presentation-v1",
        "status": "completed",
        "derivation_only": True,
        "new_llm_calls": 0,
        "new_audio_renders": 0,
        "source_result": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "metric": "Spearman rho(time, distance_to_source - distance_to_target)",
        "summary": {
            "n_style_pairs": len(rows),
            "n_plans": int(len(source["records"])),
            "n_tracks": int(source["n_tracks"]),
            "n_windows": int(source["n_windows"]),
            "pair_rho_median": float(np.median(values)),
            "pair_rho_mean": float(np.mean(values)),
            "pair_rho_mean_ci_95": bootstrap_mean_ci(values),
            "positive_pairs": int(np.sum(values > 0)),
        },
        "pair_rows": rows,
        "interpretation": {
            "positive_rho": "relative movement from source anchor toward target anchor",
            "not_claimed": "absolute target distance decreases for every pair",
            "untreated_control_included": False,
            "inferential_p_value_included": False,
        },
    }
    RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(result)
    write_table(rows)
    make_figure(rows, grouped)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(FIGURE_PNG)
    print(TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
