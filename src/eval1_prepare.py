"""Evaluation 1 (style transitions): pick the piano source windows and freeze the inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import eval1_spec as spec
from automation_fx.paths import MEDLEYDB_ARCHIVE

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(str(MEDLEYDB_ARCHIVE))
METADATA = ROOT / "vendor/medleydb/medleydb/data/Metadata"
PROMPT_SOURCE = ROOT / "demo_outputs/prompt_ablation_stage41g28/prompt_P1_original.json"
OUTPUT = ROOT / "demo_outputs/transition10_dry_piano_v1_run1"
SOURCE_DIR = OUTPUT / "source"
INPUT_DIR = OUTPUT / "dry_inputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] == 2:
        return audio
    mono = audio.mean(axis=1, keepdims=True)
    return np.repeat(mono, 2, axis=1)


def fixed_case(track_id: str, stem_id: str) -> dict:
    metadata_path = METADATA / f"{track_id}_METADATA.yaml"
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if str(document.get("has_bleed", "")).lower() != "no":
        raise RuntimeError(f"track is not bleed-free: {track_id}")
    stem = document["stems"][stem_id]
    if str(stem.get("instrument", "")).lower() != "piano":
        raise RuntimeError(f"stem is not acoustic piano: {track_id} {stem_id}")
    raw = stem.get("raw") or {}
    if len(raw) != 1:
        raise RuntimeError(f"expected one RAW component: {track_id} {stem_id}")
    raw_id, raw_row = next(iter(sorted(raw.items())))
    if str(raw_row.get("instrument", "")).lower() != "piano":
        raise RuntimeError(f"RAW is not acoustic piano: {track_id} {stem_id}")
    member = f"V1/{track_id}/{document['raw_dir']}/{raw_row['filename']}"
    return {
        "medleydb_track_id": track_id,
        "stem_id": stem_id,
        "raw_id": raw_id,
        "instrument": "piano",
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "archive_member": member,
    }


def loudest_window(audio: np.ndarray, sample_rate: int) -> tuple[int, int, list[float]]:
    length = int(round(spec.WINDOW_SECONDS * sample_rate))
    hop = int(round(spec.WINDOW_SEARCH_HOP_SECONDS * sample_rate))
    if len(audio) < length:
        raise RuntimeError(
            f"source shorter than frozen {spec.WINDOW_SECONDS:.1f}s window: {len(audio) / sample_rate:.3f}s")
    starts = list(range(0, len(audio) - length + 1, hop))
    mean_squares = [float(np.mean(np.square(audio[start:start + length]))) for start in starts]
    best_position = int(np.argmax(np.asarray(mean_squares)))
    start = starts[best_position]
    return start, start + length, mean_squares


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"output already exists; refusing to overwrite: {OUTPUT}")
    if not ARCHIVE.exists():
        raise SystemExit(f"archive missing: {ARCHIVE}")

    stored = json.loads(PROMPT_SOURCE.read_text(encoding="utf-8"))
    prompt = stored["system_prompt"]
    schema = copy.deepcopy(stored["schema"])
    schema["properties"]["target"] = {"const": "piano"}
    prompt_bundle = {"system_prompt": prompt, "schema": schema}

    cases = [fixed_case(track, stem) for track, stem in spec.PIANO_CASES]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(OUTPUT / "prompt.json", prompt_bundle)
    preregistration = {
        "stage": "transition10.dry_piano.v1",
        "run_version": spec.RUN_VERSION,
        "written_before_extraction_llm_calls_rendering_and_scoring": True,
        "input": "MedleyDB V1 single-component RAW acoustic piano",
        "case_count": len(cases),
        "fixed_cases": cases,
        "archive": str(ARCHIVE),
        "archive_sha256_from_existing_verified_selection": "574b3807b195aa05fa9395f8d29ab33b09e71345334240fee34897c70f5e31f7",
        "window_seconds": spec.WINDOW_SECONDS,
        "window_search_hop_seconds": spec.WINDOW_SEARCH_HOP_SECONDS,
        "window_rule": spec.WINDOW_RULE,
        "n_windows": spec.N_WINDOWS,
        "transition_pairs": [list(pair) for pair in spec.TRANSITION_PAIRS],
        "pair_separability": {
            f"{a}->{b}": value for (a, b), value in spec.MEASURED_PAIR_SEPARABILITY.items()
        },
        "instructions": spec.INSTRUCTIONS,
        "runs_per_instruction": spec.RUNS_PER_INSTRUCTION,
        "llm_calls_planned": len(spec.INSTRUCTIONS) * spec.RUNS_PER_INSTRUCTION,
        "model": spec.MODEL,
        "reasoning_effort": spec.REASONING_EFFORT,
        "prompt_source": str(PROMPT_SOURCE),
        "prompt_sha256": text_sha256(prompt),
        "schema_sha256": text_sha256(canonical_json(schema)),
        "schema_change_from_source": "target const only: vocals -> piano",
        "renderer_metric_and_socialfx_membership": "unchanged from transition10.v1",
        "primary_alpha": spec.PRIMARY_ALPHA,
        "conventional_alpha_reported": spec.CONVENTIONAL_ALPHA,
        "bootstrap_repetitions": spec.BOOTSTRAP_REPETITIONS,
        "known_limitations": list(spec.KNOWN_LIMITATIONS),
    }
    write_json_exclusive(OUTPUT / "PRE_REGISTRATION.json", preregistration)

    SOURCE_DIR.mkdir(parents=True, exist_ok=False)
    members = [case["archive_member"] for case in cases]
    subprocess.run(["tar", "-xzf", str(ARCHIVE), "-C", str(SOURCE_DIR), *members], check=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=False)

    prepared = []
    for case in cases:
        source_path = SOURCE_DIR / case["archive_member"]
        audio, sample_rate = sf.read(source_path, always_2d=True, dtype="float64")
        audio = to_stereo(np.asarray(audio, dtype=np.float64))
        if not np.isfinite(audio).all():
            raise RuntimeError(f"non-finite source: {source_path}")
        start, stop, candidates = loudest_window(audio, int(sample_rate))
        selected = np.ascontiguousarray(audio[start:stop])
        peak_before = float(np.max(np.abs(selected)))
        gain = min(1.0, 0.95 / peak_before) if peak_before > 0 else 1.0
        selected *= gain
        destination = INPUT_DIR / f"{case['medleydb_track_id']}__{case['stem_id']}__RAW_PIANO_16S_FLOAT.wav"
        sf.write(destination, selected.astype(np.float32), int(sample_rate), subtype="FLOAT")
        prepared.append({
            **case,
            "source_path": str(source_path),
            "source_sha256": sha256(source_path),
            "source_sample_rate": int(sample_rate),
            "source_frames": int(len(audio)),
            "source_duration_sec": float(len(audio) / sample_rate),
            "selection_start_sample": int(start),
            "selection_end_sample": int(stop),
            "selection_start_sec": float(start / sample_rate),
            "selection_end_sec": float(stop / sample_rate),
            "selection_candidate_count": len(candidates),
            "selection_max_mean_square": float(max(candidates)),
            "input_path": str(destination),
            "input_sha256": sha256(destination),
            "input_sample_rate": int(sample_rate),
            "input_frames": int(len(selected)),
            "input_duration_sec": float(len(selected) / sample_rate),
            "peak_before_safety_gain": peak_before,
            "common_safety_gain": gain,
            "peak_after_safety_gain": float(np.max(np.abs(selected))),
            "rms_after_safety_gain": float(np.sqrt(np.mean(np.square(selected)))),
        })

    write_json_exclusive(OUTPUT / "CASES.json", {
        "prepared_before_llm_calls_rendering_and_scoring": True,
        "run_version": spec.RUN_VERSION,
        "cases": prepared,
    })
    print(f"prepared {len(prepared)} dry acoustic-piano passages in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
