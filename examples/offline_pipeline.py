"""Offline run of the whole pipeline with a fake LLM and a fake encoder.

Everything else is the real code: manifest checks, endpoint construction, plan
validation, curve compilation, EQ rendering, run verification and the semantic
evaluation. Needs the Fx-Encoder++ git checkout (see README) but no API key,
no model checkpoint and no dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
import torch

import endpoint_manifest
import evaluate_run
import execute_run
import planner
import prepare_run
import reference_match
import reference_matcher
import semantic_case
import text_endpoints

SAMPLE_RATE = 44_100
INSTRUCTION = "start warm then slowly move to the reference sound and hold it"


def _write_inputs(folder: Path) -> Path:
    t = np.arange(SAMPLE_RATE * 3, dtype=np.float32) / SAMPLE_RATE
    target = np.stack((0.015 * np.sin(2 * np.pi * 227 * t), 0.013 * np.sin(2 * np.pi * 263 * t)), 1)
    tr = np.arange(execute_run.REFERENCE_CLIP_FRAMES, dtype=np.float32) / SAMPLE_RATE
    reference = np.stack((0.011 * np.sin(2 * np.pi * 311 * tr), 0.009 * np.sin(2 * np.pi * 347 * tr)), 1)
    sf.write(folder / "target.wav", target.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    sf.write(folder / "reference.wav", reference.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    structure = {
        "bpm": 120.0,
        "beats": [0.1, 0.55, 1.0, 1.45, 1.9, 2.35, 2.8],
        "downbeats": [0.1, 1.0, 1.9, 2.8],
        "sections": [{"label": "unit", "occurrence": 1, "start_sec": 0.1, "end_sec": 2.8}],
    }
    (folder / "structure.json").write_text(json.dumps(structure))
    manifest = {
        "schema_version": endpoint_manifest.SCHEMA_VERSION,
        "target": {
            "raw_stem_path": str(folder / "target.wav"),
            "structure_path": str(folder / "structure.json"),
            "stem": "vocals",
            "section": "unit",
            "occurrence": 1,
        },
        "automation_instruction": INSTRUCTION,
        "renderer_mode": "eq",
        "endpoints": [
            {"kind": "natural_language", "text": "a warm dark vocal", "effect_mode": "eq"},
            {"kind": "processed_reference", "audio_path": str(folder / "reference.wav"), "effect_mode": "eq"},
        ],
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _freeze_semantic_case(prepared: Path, output: Path) -> None:
    semantic_case.prepare_case(
        case_id="offline_case",
        prepared_path=prepared,
        output=output,
        evidence_class="overlap_or_direct_cue_smoke",
        scored_endpoint_ordinals=[1],
        expected_route={
            "initial_ordinal": 0,
            "initial_evidence_exact": "start warm",
            "transition": {
                "source_ordinal": 0,
                "target_ordinal": 1,
                "instruction_evidence_exact": "slowly move to the reference sound",
            },
            "exact_event_count": None,
            "event_count_evidence_exact": None,
            "final_ordinal": None,
            "final_evidence_exact": None,
        },
        qualitative_curve_constraints=[
            {
                "trend": "monotonic_toward_target",
                "completion": "unspecified",
                "instruction_evidence_exact": "slowly move to the reference sound",
            }
        ],
        metric_config={
            "activity_rms_dbfs_threshold": -50.0,
            "delta_presence_relative_db_threshold": -30.0,
            "minimum_active_bars": 1,
            "threshold_rationale": "Frozen signal proxy; no audibility claim.",
            "human_audibility_claimed": False,
        },
        annotation_protocol={
            "annotator_count": 2,
            "agreement_frozen_before_execution": True,
            "direct_cue_reviewer_ids": ["reviewer_alpha", "reviewer_beta"],
            "direct_cue_review_completed_before_execution": True,
            "direct_cue_review_conclusion": "smoke_only_direct_cue_overlap_or_design_blocker",
            "direct_cue_review_rationale": (
                "Fixture contains fewer than two natural-language alternatives, so "
                "selection evidence is conservatively smoke-only."
            ),
            "protocol_note": "Independent annotation before execution preparation.",
        },
    )


def _static_plan(payload: dict) -> dict:
    schema = payload["response_format"]["json_schema"]["schema"]
    ranges = reference_matcher.official_eq_parameter_ranges()
    gains = [3.0, 1.5, -1.0, -2.5, -4.0, -6.0]
    eq = {}
    for index, band in enumerate(text_endpoints.EQ_BANDS):
        frequency = ranges[f"{band}_cutoff_freq"]
        q_factor = ranges[f"{band}_q_factor"]
        eq[band] = {
            "gain_db": gains[index % len(gains)],
            "frequency_hz": 0.5 * (frequency[0] + frequency[1]),
            "q_factor": 0.5 * (q_factor[0] + q_factor[1]),
        }
    return {
        "schema_version": text_endpoints.STATIC_PLAN_VERSION,
        "target_stem": schema["properties"]["target_stem"]["const"],
        "effect_mode": schema["properties"]["effect_mode"]["const"],
        "parameters": {"eq": eq},
    }


def _event(event_id, evidence, relation, source, target, amount, duration):
    return {
        "id": event_id,
        "instruction_evidence": evidence,
        "relation": relation,
        "source_endpoint_id": source,
        "target_endpoint_id": target,
        "target_amount": amount,
        "duration_weight": duration,
        "handle1_time_fraction": 0.3,
        "handle2_time_fraction": 0.7,
        "handle1_value_fraction": 0.1,
        "handle2_value_fraction": 0.9,
    }


def _automation_plan(payload: dict) -> dict:
    properties = payload["response_format"]["json_schema"]["schema"]["properties"]
    return {
        "schema_version": planner.SCHEMA_VERSION,
        "target_stem": properties["target_stem"]["const"],
        "effect_mode": properties["effect_mode"]["const"],
        "section": properties["section"]["const"],
        "occurrence": properties["occurrence"]["const"],
        "initial_endpoint_id": "endpoint_000",
        "events": [
            _event("e1", "slowly move to the reference sound", "set_state", "endpoint_000", "endpoint_001", 1.0, 2.0),
            _event("e2", "hold it", "hold_current", None, None, None, 1.0),
        ],
    }


def _fake_llm(calls: list):
    def call(payload, *, base_url, timeout, reasoning_effort):
        calls.append(payload)
        plan = _static_plan(payload) if len(calls) == 1 else _automation_plan(payload)
        response = {"choices": [{"message": {"content": json.dumps(plan)}}], "native_meta": {"finish_reason": "stop"}}
        request = execute_run._native_request(payload, base_url=base_url, reasoning_effort=reasoning_effort)
        return response, request

    return call


def _fake_match(encoder, reference_audio, target_audio, mode, *, renderer, learning_rate, initial_normalized=None):
    parameters = torch.full(
        (1, reference_matcher.PARAM_COUNTS[mode]), 0.5, dtype=target_audio.dtype, device=target_audio.device
    )
    parameters[0, ::3] = 0.8
    rendered = renderer.render(target_audio, parameters, mode).detach().clone()
    return reference_matcher.MatchResult(
        mode=mode,
        initial_normalized_parameters=parameters.clone(),
        normalized_parameters=parameters,
        physical_parameters={},
        rendered_audio=rendered,
        loss_trace=tuple([0.25] * (reference_matcher.MATCH_STEPS + 1)),
    )


def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(tempfile.mkdtemp())
    checkpoint = cache / "models--yytung--fxencoder-plusplus" / "snapshots" / "offline"
    checkpoint.mkdir(parents=True)
    (checkpoint / reference_match.FXENCODER_CHECKPOINT_NAME).write_bytes(b"placeholder: the encoder is faked")
    reference_match.FXENCODER_CACHE_ROOT = cache

    manifest = _write_inputs(output)
    prepared = output / "prepared"
    prepare_run.prepare(manifest_path=manifest, output=prepared)
    prepare_run.verify_prepared(prepared)

    semantic = output / "semantic_case"
    _freeze_semantic_case(prepared, semantic)

    execution = output / "execution"
    execute_run._prepare_test_execution(prepared_path=prepared, semantic_prepared=semantic, output=execution)

    calls: list = []
    dependencies = execute_run._make_test_runtime_dependencies(
        load_encoder=lambda record: torch.nn.Identity(), match_reference=_fake_match
    )
    run_dir = output / "run"
    execute_run._execute_for_test(
        prepared=execution,
        output=run_dir,
        reviewed=True,
        test_double=_fake_llm(calls),
        runtime_dependencies=dependencies,
        timeout=1.0,
    )
    execute_run._verify_completed_test_run(run_dir)
    evaluation = evaluate_run.evaluate_completed_case(semantic_prepared=semantic, completed_run=run_dir)
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=1, sort_keys=True, default=str))
    return {"llm_calls": len(calls), "run": str(run_dir), "rendered": str(run_dir / "target_after_FLOAT.wav")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=1))


if __name__ == "__main__":
    main()
