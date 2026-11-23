"""Evaluation 1 (style transitions): render the planned curves and score their direction."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from automation_fx.paths import REPO_ROOT

ROOT = Path(__file__).resolve().parent
PROJECT = REPO_ROOT

from datasets import load_dataset

import eval1_spec as spec
import signed_curve as curve
import socialfx
import style_eq
import trajectory_score
from automation_fx.audio import peak_report

OUTPUT = ROOT / "demo_outputs/transition10_dry_piano_v1_run1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrected_curves() -> dict[str, list[np.ndarray]]:
    gen_eval = load_dataset("seungheondoh/socialfx-gen-eval", split="eq")
    original = load_dataset("seungheondoh/socialfx-original", split="eq")
    by_id = {str(row["id"]): row for row in original}
    curves = {}
    for word in spec.WORDS:
        rows = [row for row in gen_eval if str(row["input"]) == word]
        if len(rows) != 1:
            raise RuntimeError(f"SocialFX membership changed for {word}")
        curves[word] = [
            np.asarray(by_id[str(value)]["param_values"], dtype=float)
            for value in rows[0]["output"]
        ]
    counts = {word: len(values) for word, values in curves.items()}
    if counts != spec.EXPECTED_SOCIALFX_COUNTS:
        raise RuntimeError(f"SocialFX counts changed: {counts}")
    return curves


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    started = time.perf_counter()
    result_path = OUTPUT / "result.json"
    if result_path.exists():
        raise SystemExit("piano result already exists; refusing to overwrite")
    prereg = json.loads((OUTPUT / "PRE_REGISTRATION.json").read_text(encoding="utf-8"))
    prepared = json.loads((OUTPUT / "CASES.json").read_text(encoding="utf-8"))
    plan_summary = json.loads((OUTPUT / "plan_summary.json").read_text(encoding="utf-8"))
    if prereg["run_version"] != spec.RUN_VERSION:
        raise RuntimeError("piano pre-registration version mismatch")
    if plan_summary["completed"] != 30 or plan_summary["failed"] != 0:
        raise RuntimeError(f"incomplete piano plans: {plan_summary}")
    if plan_summary["retry_repair_selection"] != 0:
        raise RuntimeError("piano plan generation was not one-shot")

    songs = []
    for case in prepared["cases"]:
        path = Path(case["input_path"])
        if sha256(path) != case["input_sha256"]:
            raise RuntimeError(f"prepared input changed: {case['medleydb_track_id']}")
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
        if int(sample_rate) != int(case["input_sample_rate"]):
            raise RuntimeError(f"sample rate changed: {case['medleydb_track_id']}")
        if int(len(audio)) != int(case["input_frames"]):
            raise RuntimeError(f"frame count changed: {case['medleydb_track_id']}")
        if not np.isfinite(audio).all() or float(np.sqrt(np.mean(np.square(audio)))) <= 1e-5:
            raise RuntimeError(f"invalid or inactive piano input: {case['medleydb_track_id']}")
        songs.append({"case": case, "source": np.ascontiguousarray(audio), "sample_rate": int(sample_rate)})
    sample_rates = {song["sample_rate"] for song in songs}
    frame_counts = {len(song["source"]) for song in songs}
    if len(sample_rates) != 1 or len(frame_counts) != 1:
        raise RuntimeError(f"piano inputs are not aligned: rates={sample_rates}, frames={frame_counts}")
    sample_rate = next(iter(sample_rates))
    n_samples = next(iter(frame_counts))
    if n_samples != int(round(spec.WINDOW_SECONDS * sample_rate)):
        raise RuntimeError("piano passage duration changed")

    curves = corrected_curves()
    print(f"dry acoustic piano: {len(songs)} tracks, {spec.N_WINDOWS} windows", flush=True)
    print("building SocialFX reference distributions on the same dry piano inputs", flush=True)
    reference = socialfx.build_reference(
        [song["source"] for song in songs], curves, sample_rate, spec.N_WINDOWS)

    plan_paths = sorted((OUTPUT / "runs").glob("*/raw_response.txt"))
    if len(plan_paths) != 30:
        raise RuntimeError(f"expected 30 raw plans, found {len(plan_paths)}")
    records = []
    for plan_path in plan_paths:
        name = plan_path.parent.name
        key = name.rsplit("__run", 1)[0]
        if key not in spec.INSTRUCTIONS:
            raise RuntimeError(f"unknown piano plan: {name}")
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        if raw.get("target") != "piano" or raw.get("effect") != "eq":
            raise RuntimeError(f"wrong routing in piano plan: {name}")
        axis, _ = style_eq.decode_axis(raw["eq_axis"])
        parameters = style_eq.parameters_for(
            axis, float(raw["max_axis_gain_db"]), sample_rate, raw.get("band_shape"))
        typed, _ = curve.resolve_signed_curve(raw["trajectory"], instruction=None)
        full_curve = np.asarray(
            curve.evaluate_signed_curve(typed, n_samples)["values"], dtype=float)
        windows = socialfx.equal_windows(n_samples, spec.N_WINDOWS)
        planned = np.asarray([
            np.mean(full_curve[start:stop]) for start, stop in windows
        ], dtype=float)
        initial = float(raw["trajectory"]["initial_level"])
        final = float(raw["trajectory"]["events"][-1]["level"])
        denominator = final - initial
        if abs(denominator) < 1e-12:
            raise RuntimeError(f"zero endpoint span in {name}")
        planned_progress = (planned - initial) / denominator

        features = []
        clipping_samples = 0
        for song in songs:
            rendered, _ = style_eq.render_signed_curve(
                torch.as_tensor(song["source"], dtype=torch.float64),
                torch.as_tensor(full_curve, dtype=torch.float64),
                parameters,
            )
            processed = np.ascontiguousarray(rendered.detach().cpu().numpy())
            clipping_samples += int(peak_report(processed)["samples_over_1.0"])
            mono = processed.mean(axis=1)
            features.append([
                socialfx._features(mono, start, stop, sample_rate)
                for start, stop in windows
            ])
        feature_array = np.asarray(features)
        distances = {
            word: np.asarray([
                [
                    socialfx.standardized_distance(
                        feature_array[track, window], values[:, window, :])
                    for window in range(spec.N_WINDOWS)
                ]
                for track in range(len(songs))
            ])
            for word, values in reference.items()
        }
        expected = spec.EXPECTED[key]
        scoring = trajectory_score.score_run(
            distances, expected["source_word"], expected["target_word"])
        observed = distances[expected["source_word"]] - distances[expected["target_word"]]
        plan_audio_track = [
            trajectory_score.rank_corr(planned_progress, observed[index])
            for index in range(len(songs))
        ]
        plan_audio_track = [float(value) for value in plan_audio_track if value is not None]
        pooled_plan_audio = trajectory_score.rank_corr(
            planned_progress, np.mean(observed, axis=0))
        per_track_scores = [
            row["score"] for row in scoring["per_track"] if row["score"] is not None
        ]
        positive_tracks = sum(float(value) > 0 for value in per_track_scores)
        record = {
            "instruction_key": key,
            "run": name,
            "source_word": expected["source_word"],
            "target_word": expected["target_word"],
            "separability": expected["separability"],
            "clipping_samples": clipping_samples,
            "max_axis_gain_db": float(raw["max_axis_gain_db"]),
            "curve_initial_level": initial,
            "curve_final_level": final,
            "scoring": {
                **scoring,
                "pass_primary_p_le_0_10": bool(
                    scoring["pooled"]["p_value"] is not None
                    and scoring["pooled"]["p_value"] <= spec.PRIMARY_ALPHA),
                "pass_conventional_p_le_0_05": bool(
                    scoring["pooled"]["p_value"] is not None
                    and scoring["pooled"]["p_value"] <= spec.CONVENTIONAL_ALPHA),
                "directional_majority_pass": bool(
                    scoring["pooled"]["score"] is not None
                    and scoring["pooled"]["score"] > 0
                    and scoring["score_median"] is not None
                    and scoring["score_median"] > 0
                    and positive_tracks >= (len(songs) // 2 + 1)),
                "positive_tracks": positive_tracks,
                "plan_audio_pooled_rho": float(pooled_plan_audio),
                "plan_audio_median_track_rho": float(np.median(plan_audio_track)),
            },
            "distances": {word: values.tolist() for word, values in distances.items()},
        }
        records.append(record)
        print(
            f"{name:28s} rho={scoring['score_median']:+.3f} "
            f"pooled={scoring['pooled']['score']:+.3f} "
            f"p={scoring['pooled']['p_value']:.4f}",
            flush=True,
        )

    grouped = defaultdict(list)
    for record in records:
        grouped[record["instruction_key"]].append(record)
    pair_rows = []
    for key in spec.INSTRUCTION_KEYS:
        members = grouped[key]
        if len(members) != spec.RUNS_PER_INSTRUCTION:
            raise RuntimeError(f"missing runs for {key}")
        pair_rows.append({
            "style_pair": key,
            "n_runs": len(members),
            "median_track_rho_over_runs": float(np.median([
                row["scoring"]["score_median"] for row in members
            ])),
            "median_pooled_rho_over_runs": float(np.median([
                row["scoring"]["pooled"]["score"] for row in members
            ])),
            "median_plan_audio_rho_over_runs": float(np.median([
                row["scoring"]["plan_audio_median_track_rho"] for row in members
            ])),
            "primary_pass_runs": int(sum(
                row["scoring"]["pass_primary_p_le_0_10"] for row in members)),
            "conventional_pass_runs": int(sum(
                row["scoring"]["pass_conventional_p_le_0_05"] for row in members)),
            "directional_pass_runs": int(sum(
                row["scoring"]["directional_majority_pass"] for row in members)),
        })

    rng = np.random.default_rng(spec.BOOTSTRAP_SEED)
    boot_rho = np.empty(spec.BOOTSTRAP_REPETITIONS)
    boot_plan_audio = np.empty(spec.BOOTSTRAP_REPETITIONS)
    for index in range(spec.BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, len(pair_rows), len(pair_rows))
        sampled = [pair_rows[position] for position in chosen]
        boot_rho[index] = np.mean([
            row["median_track_rho_over_runs"] for row in sampled
        ])
        boot_plan_audio[index] = np.mean([
            row["median_plan_audio_rho_over_runs"] for row in sampled
        ])

    summary = {
        "n_style_pairs": len(pair_rows),
        "n_runs": len(records),
        "n_tracks": len(songs),
        "primary_pass_runs_p_le_0_10": int(sum(
            row["scoring"]["pass_primary_p_le_0_10"] for row in records)),
        "conventional_pass_runs_p_le_0_05": int(sum(
            row["scoring"]["pass_conventional_p_le_0_05"] for row in records)),
        "directional_majority_pass_runs": int(sum(
            row["scoring"]["directional_majority_pass"] for row in records)),
        "median_track_rho_over_all_runs": float(np.median([
            row["scoring"]["score_median"] for row in records
        ])),
        "median_pooled_rho_over_all_runs": float(np.median([
            row["scoring"]["pooled"]["score"] for row in records
        ])),
        "median_plan_audio_track_rho_over_all_runs": float(np.median([
            row["scoring"]["plan_audio_median_track_rho"] for row in records
        ])),
        "pairs_with_positive_median_rho": int(sum(
            row["median_track_rho_over_runs"] > 0 for row in pair_rows)),
        "mean_pair_median_rho_95pct_ci": percentile_interval(boot_rho),
        "mean_pair_plan_audio_rho_95pct_ci": percentile_interval(boot_plan_audio),
        "total_clipping_samples": int(sum(row["clipping_samples"] for row in records)),
    }
    result = {
        "stage": "transition10.dry_piano.v1",
        "run_version": spec.RUN_VERSION,
        "llm_calls_during_scoring": 0,
        "input": "MedleyDB V1 single-component RAW acoustic piano, activity-selected 16-second passages",
        "n_tracks": len(songs),
        "n_windows": spec.N_WINDOWS,
        "tracks": [song["case"]["medleydb_track_id"] for song in songs],
        "stem_ids": [song["case"]["stem_id"] for song in songs],
        "socialfx_counts": {word: len(values) for word, values in curves.items()},
        "summary": summary,
        "pair_rows": pair_rows,
        "records": records,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json_exclusive(result_path, result)

    lines = [
        "# Dry acoustic piano transition evaluation",
        "",
        f"- Inputs: {len(songs)} MedleyDB V1 RAW acoustic-piano passages",
        f"- Plans: {len(records)} one-shot LLM plans across {len(pair_rows)} ordered style pairs",
        f"- Primary p <= 0.10: {summary['primary_pass_runs_p_le_0_10']}/{len(records)}",
        f"- Conventional p <= 0.05: {summary['conventional_pass_runs_p_le_0_05']}/{len(records)}",
        f"- Directional-majority: {summary['directional_majority_pass_runs']}/{len(records)}",
        f"- Median per-track rho: {summary['median_track_rho_over_all_runs']:.3f}",
        f"- Median pooled rho: {summary['median_pooled_rho_over_all_runs']:.3f}",
        f"- Positive style pairs: {summary['pairs_with_positive_median_rho']}/{len(pair_rows)}",
        "",
        "| Style pair | Median track rho | p<=.10 | p<=.05 | Directional |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['style_pair']} | {row['median_track_rho_over_runs']:.3f} | "
            f"{row['primary_pass_runs']}/3 | {row['conventional_pass_runs']}/3 | "
            f"{row['directional_pass_runs']}/3 |"
        )
    (OUTPUT / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
