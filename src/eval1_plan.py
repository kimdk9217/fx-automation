"""Evaluation 1 (style transitions): ask the planner for a curve for each transition instruction."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import jsonschema
import numpy as np
import soundfile as sf

from automation_fx.paths import REPO_ROOT

ROOT = Path(__file__).resolve().parent
PROJECT = REPO_ROOT

import eval1_spec as spec
import llm
import signed_curve as curve
import style_eq

OUTPUT = ROOT / "demo_outputs/transition10_dry_piano_v1_run1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    started = time.perf_counter()
    prereg = json.loads((OUTPUT / "PRE_REGISTRATION.json").read_text(encoding="utf-8"))
    cases = json.loads((OUTPUT / "CASES.json").read_text(encoding="utf-8"))
    prompt_bundle = json.loads((OUTPUT / "prompt.json").read_text(encoding="utf-8"))
    if prereg["run_version"] != spec.RUN_VERSION:
        raise RuntimeError("piano pre-registration version mismatch")
    if not cases["prepared_before_llm_calls_rendering_and_scoring"]:
        raise RuntimeError("piano inputs were not prepared before planning")
    if len(cases["cases"]) != len(spec.PIANO_CASES):
        raise RuntimeError("piano case count changed")
    schema = prompt_bundle["schema"]
    prompt = prompt_bundle["system_prompt"]
    schema_hash = hashlib.sha256(json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if schema_hash != prereg["schema_sha256"]:
        raise RuntimeError("frozen piano schema changed")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prereg["prompt_sha256"]:
        raise RuntimeError("frozen system prompt changed")

    representative = cases["cases"][0]
    audio, sample_rate = sf.read(
        representative["input_path"], always_2d=True, dtype="float64")
    if sha256(Path(representative["input_path"])) != representative["input_sha256"]:
        raise RuntimeError("representative piano input changed")
    source_state = style_eq.source_context(np.ascontiguousarray(audio), int(sample_rate))
    n_samples = int(len(audio))
    duration = float(n_samples / sample_rate)
    timing = {
        "section": "selected piano passage",
        "occurrence": 1,
        "raw_start_sec": 0.0,
        "raw_end_sec": duration,
        "aligned_start_sec": 0.0,
        "aligned_end_sec": duration,
        "downbeats_sec": [],
        "bar_count": None,
        "render_start_sample": 0,
        "active_start_sample": 0,
        "active_end_sample": n_samples,
        "render_end_sample": n_samples,
        "render_duration_sec": duration,
        "active_duration_sec": duration,
        "context_before_sec": 0.0,
        "context_after_sec": 0.0,
    }

    runs_directory = OUTPUT / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    completed = failed = 0
    for key in spec.INSTRUCTION_KEYS:
        instruction = spec.INSTRUCTIONS[key]
        for attempt in range(1, spec.RUNS_PER_INSTRUCTION + 1):
            name = f"{key}__run{attempt}"
            directory = runs_directory / name
            raw_path = directory / "raw_response.txt"
            failure_path = directory / "failure.json"
            if failure_path.exists():
                failed += 1
                continue
            if raw_path.exists():
                completed += 1
                continue
            directory.mkdir(parents=True, exist_ok=True)
            context = {
                "RUN_VERSION": spec.RUN_VERSION,
                "INSTRUCTION": instruction,
                "LOCKED_ROUTING": {
                    "target": "piano",
                    "effect": "eq",
                    "section": "selected piano passage",
                    "occurrence": 1,
                    "timing": timing,
                    "filter_effect_allowed": False,
                },
                "SOURCE_AUDIO_STATE": source_state,
                "CURRENT_SESSION_EQ_STATE": {
                    "new_eq_node": True,
                    "neutral_state": 0.0,
                    "upstream_baked_parameters": "unknown and not inferred",
                },
            }
            payload = {
                "model": spec.MODEL,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "signed_eq_freq_q_piano",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            write_json_exclusive(directory / "llm_request.json", {
                "payload": payload,
                "context": context,
                "representative_input_path": representative["input_path"],
                "representative_input_sha256": representative["input_sha256"],
            })
            try:
                response, _ = llm.call_openai_structured_once(
                    payload, timeout=900.0, reasoning_effort=spec.REASONING_EFFORT)
                response_text = response["choices"][0]["message"]["content"]
                raw_path.write_text(response_text, encoding="utf-8")
                raw = json.loads(response_text)
                jsonschema.Draft202012Validator(schema).validate(raw)
                events = raw["trajectory"]["events"]
                if not events:
                    raise ValueError("transition instruction produced no event")
                if abs(float(events[-1]["level"]) - float(raw["trajectory"]["initial_level"])) < 1e-12:
                    raise ValueError("transition instruction produced identical start and end levels")
                axis, _ = style_eq.decode_axis(raw["eq_axis"])
                style_eq.parameters_for(
                    axis, float(raw["max_axis_gain_db"]), int(sample_rate), raw.get("band_shape"))
                curve.resolve_signed_curve(raw["trajectory"], instruction=None)
                completed += 1
                print(f"[{name}] ok max_gain={raw['max_axis_gain_db']}", flush=True)
            except Exception as error:
                failed += 1
                write_json_exclusive(failure_path, {
                    "instruction_key": key,
                    "attempt": attempt,
                    "retry_performed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                })
                print(f"[{name}] FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    write_json_exclusive(OUTPUT / "plan_summary.json", {
        "completed": completed,
        "failed": failed,
        "llm_calls_planned": len(spec.INSTRUCTIONS) * spec.RUNS_PER_INSTRUCTION,
        "retry_repair_selection": 0,
        "elapsed_sec": time.perf_counter() - started,
    })
    print(f"completed {completed}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
