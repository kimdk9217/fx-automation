"""Checks a finished run against its frozen semantic case and computes the verdicts."""

from __future__ import annotations

import json
import math
import os
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

import endpoint_manifest as manifest
import execute_run
import semantic_case as semantic_prereg

EVALUATOR_VERSION = "dynamic-multi-endpoint-semantic-evaluation-v3"


class EvaluateRunError(ValueError):
    pass


def _file_identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise EvaluateRunError(
            f"missing frozen file: {path}"
        )
    return {"path": str(path), "sha256": manifest.file_sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise EvaluateRunError(
            f"invalid JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EvaluateRunError(
            f"expected JSON object: {path}"
        )
    return value


def _walk_forbidden_keys(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        if forbidden.intersection(str(key) for key in value):
            return True
        return any(_walk_forbidden_keys(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(_walk_forbidden_keys(child, forbidden) for child in value)
    return False


def _verify_source_and_hash_chain(
    *,
    semantic_path: Path,
    semantic_case: Mapping[str, Any],
    semantic_digest: str,
    run: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_prereg = _read_json(run / "PRE_REGISTRATION.json")
    plan_source = run_prereg.get("plan_source")
    if plan_source == execute_run.LIVE_PLAN_SOURCE:
        verification = execute_run.verify_completed_run(run)
    elif plan_source == execute_run.TEST_PLAN_SOURCE:
        verification = execute_run._verify_completed_test_run(run)
    else:
        raise EvaluateRunError(
            "completed run has an unknown plan source"
        )
    if verification.get("verified") is not True or verification.get(
        "plan_source"
    ) != plan_source:
        raise EvaluateRunError(
            "completed-run verifier did not confirm the derived plan source"
        )

    code = run_prereg.get("reproducibility", {}).get("code", {})
    frozen_evaluator = (
        code.get("semantic_postrun_evaluator")
        if isinstance(code, Mapping)
        else None
    )
    current_evaluator = _file_identity(Path(__file__))
    if frozen_evaluator != current_evaluator:
        raise EvaluateRunError(
            "post-run evaluator was not hash-frozen in the pre-API v4 execution preregistration"
        )

    result = _read_json(run / "result.json")
    completed = _read_json(run / "COMPLETED.json")
    identity = run_prereg.get("semantic_evaluation_pre_registration")
    if not isinstance(identity, Mapping):
        raise EvaluateRunError(
            "execution preregistration lost semantic identity"
        )
    bound_directory = Path(str(identity.get("directory", ""))).resolve()
    source_binding = semantic_case.get("source_binding")
    execution_source = run_prereg.get("source_v3_prepared")
    copied = run / "semantic_pre_registration"
    expected_copy_files = {
        "SEMANTIC_PRE_REGISTRATION.json",
        "SEMANTIC_PRE_REGISTRATION.sha256",
        "PREPARATION_STATUS.json",
    }
    if (
        bound_directory != semantic_path
        or run_prereg.get("semantic_pre_registration_sha256") != semantic_digest
        or verification.get("pre_registration_sha256")
        != result.get("pre_registration_sha256")
        or result.get("semantic_pre_registration_sha256") != semantic_digest
        or completed.get("semantic_pre_registration_sha256") != semantic_digest
        or not isinstance(source_binding, Mapping)
        or not isinstance(execution_source, Mapping)
        or Path(str(execution_source.get("directory", ""))).resolve()
        != Path(str(source_binding.get("v3_prepared_directory", ""))).resolve()
        or execution_source.get("verified_pre_registration_sha256")
        != source_binding.get("v3_pre_registration_sha256")
        or run_prereg.get("canonical_manifest_sha256")
        != source_binding.get("canonical_manifest_sha256")
        or not copied.is_dir()
        or {path.name for path in copied.iterdir()} != expected_copy_files
    ):
        raise EvaluateRunError(
            "semantic -> execution -> completed-run hash chain changed"
        )

    identity_keys = {
        "SEMANTIC_PRE_REGISTRATION.json": "pre_registration",
        "SEMANTIC_PRE_REGISTRATION.sha256": "sidecar",
        "PREPARATION_STATUS.json": "preparation_status",
    }
    for filename, identity_key in identity_keys.items():
        frozen_identity = identity.get(identity_key)
        if (
            not isinstance(frozen_identity, Mapping)
            or manifest.file_sha256(semantic_path / filename)
            != frozen_identity.get("sha256")
            or manifest.file_sha256(copied / filename)
            != frozen_identity.get("sha256")
        ):
            raise EvaluateRunError(
                f"semantic preregistration copy or identity changed: {filename}"
            )
    if _read_json(copied / "SEMANTIC_PRE_REGISTRATION.json") != semantic_case:
        raise EvaluateRunError(
            "copied semantic preregistration content changed"
        )

    binding_report = {
        "plan_source_derived_from_verified_execution_artifact": plan_source,
        "caller_supplied_plan_source": False,
        "semantic_preregistration_bound_into_execution_preregistration": True,
        "semantic_preregistration_reverified_by_v4_before_saved_model_calls": True,
        "semantic_bundle_copied_and_hash_bound_into_completed_run": True,
        "semantic_pre_registration_sha256": semantic_digest,
        "execution_pre_registration_sha256": verification[
            "pre_registration_sha256"
        ],
        "result_sha256": verification["result_sha256"],
        "postrun_evaluator_code_frozen_before_api": True,
        "postrun_evaluator_code": current_evaluator,
        "external_timestamp_or_remote_attestation_claimed": False,
    }
    return run_prereg, result, verification, binding_report


def _verify_request_blinding(
    *, case: Mapping[str, Any], run: Path, expected_calls: int
) -> dict[str, Any]:
    requests = sorted(run.rglob("request.json"))
    responses = sorted(run.rglob("raw_response_before_parse.json"))
    if len(requests) != expected_calls or len(responses) != expected_calls:
        raise EvaluateRunError(
            "saved request/response count differs from the frozen API-call budget"
        )
    forbidden = {
        "case_nonce",
        "expected_route",
        "qualitative_curve_constraints",
        "metric_config",
        "annotation_protocol",
        "selection_capability_claim_policy",
        "visibility_contract",
    }
    for path in requests:
        record = _read_json(path)
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or _walk_forbidden_keys(payload, forbidden):
            raise EvaluateRunError(
                "evaluator-only semantic material reached a saved model request"
            )
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        for secret in (
            case.get("case_nonce"),
            case.get("case_id"),
            case.get("evaluator_version"),
        ):
            if isinstance(secret, str) and secret and secret in serialized:
                raise EvaluateRunError(
                    "semantic evaluator identity reached a saved model request"
                )
    return {
        "execution_preregistration_hash_chain_not_mtime_used": True,
        "saved_request_count_inspected": len(requests),
        "saved_response_count_inspected": len(responses),
        "evaluator_only_route_curve_threshold_annotation_nonce_absent": True,
        "caller_instruction_and_endpoint_texts_visible_by_design": True,
        "evaluator_api_calls": 0,
    }


def _native_response_provenance(
    *,
    case: Mapping[str, Any],
    run: Path,
    run_prereg: Mapping[str, Any],
    plan_source: str,
    expected_calls: int,
) -> dict[str, Any]:
    response_paths = sorted(run.rglob("raw_response_before_parse.json"))
    requested_model = str(run_prereg["endpoint_generation_config"]["model"])
    frozen_second = int(case["frozen_at_unix_ns"]) // 1_000_000_000
    rows: list[dict[str, Any]] = []
    complete = len(response_paths) == expected_calls
    for path in response_paths:
        wrapper = _read_json(path)
        response = wrapper.get("response")
        meta = response.get("native_meta") if isinstance(response, Mapping) else None
        valid = isinstance(meta, Mapping)
        model = meta.get("model") if valid else None
        created = meta.get("created") if valid else None
        finish = meta.get("finish_reason") if valid else None
        prompt = meta.get("prompt_tokens") if valid else None
        completion = meta.get("completion_tokens") if valid else None
        total = meta.get("total_tokens") if valid else None
        integers = (prompt, completion, total)
        token_usage_valid = bool(
            all(
                not isinstance(value, bool)
                and isinstance(value, int)
                and value >= 0
                for value in integers
            )
            and int(total) == int(prompt) + int(completion)
        )
        model_matches_request = bool(
            isinstance(model, str)
            and model
            and (model == requested_model or model.startswith(requested_model + "-"))
        )
        created_valid = bool(
            not isinstance(created, bool)
            and isinstance(created, int)
            and created >= frozen_second
        )
        row_valid = bool(
            valid
            and finish == "stop"
            and model_matches_request
            and created_valid
            and token_usage_valid
        )
        complete = bool(complete and row_valid)
        rows.append(
            {
                "artifact": str(path.relative_to(run)),
                "reported_model": model,
                "requested_model": requested_model,
                "model_matches_frozen_request": model_matches_request,
                "created": created,
                "created_not_before_semantic_freeze": created_valid,
                "finish_reason": finish,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "nonnegative_consistent_token_usage": token_usage_valid,
                "complete": row_valid,
            }
        )
    live = plan_source == execute_run.LIVE_PLAN_SOURCE
    return {
        "plan_source": plan_source,
        "live_entrypoint": live,
        "expected_response_count": expected_calls,
        "native_response_metadata_complete": complete,
        "responses": rows,
        "required_for_live_case_level_model_evidence": True,
        "test_double_metadata_never_upgrades_to_live": True,
        "cryptographic_remote_attestation": False,
    }
def _ordinal_ids_from_verified_result(
    case: Mapping[str, Any], result: Mapping[str, Any]
) -> list[str]:
    records = result.get("endpoint_records")
    endpoints = case.get("endpoints")
    if (
        not isinstance(endpoints, list)
        or not isinstance(records, list)
        or len(records) != len(endpoints) + 1
    ):
        raise EvaluateRunError(
            "verified endpoint records cannot define the ordinal map"
        )
    identifiers: list[str] = []
    for ordinal, row in enumerate(records):
        if not isinstance(row, Mapping) or not isinstance(row.get("endpoint_id"), str):
            raise EvaluateRunError(
                "verified endpoint record shape changed"
            )
        if ordinal == 0:
            if row.get("role") != "dry" or row.get("endpoint_mode") is not None:
                raise EvaluateRunError(
                    "ordinal zero is not the verified dry endpoint"
                )
        else:
            expected = endpoints[ordinal - 1]
            expected_role = (
                "caller_natural_language"
                if expected["kind"] == "natural_language"
                else "processed_reference"
            )
            if (
                row.get("role") != expected_role
                or row.get("endpoint_mode") != expected["effect_mode"]
            ):
                raise EvaluateRunError(
                    "completed endpoint ordinal role or effect mode changed"
                )
            if expected_role == "caller_natural_language" and (
                row.get("caller_description_exact")
                != expected["caller_text_exact"]
                or row.get("caller_description_utf8_sha256")
                != expected["caller_text_utf8_sha256"]
            ):
                raise EvaluateRunError(
                    "completed endpoint caller text binding changed"
                )
        identifiers.append(str(row["endpoint_id"]))
    if len(identifiers) != len(set(identifiers)):
        raise EvaluateRunError(
            "completed endpoint IDs are not unique"
        )
    return identifiers


def _evaluate_route_and_curve(
    *,
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    ordinal_ids: Sequence[str],
    weights: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    automation = result.get("automation")
    if not isinstance(automation, Mapping):
        raise EvaluateRunError(
            "validated automation result disappeared"
        )
    plan = automation.get("validated_plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("events"), list):
        raise EvaluateRunError(
            "validated plan events disappeared"
        )
    events = plan["events"]
    expected = case["expected_route"]
    transition = expected["transition"]
    expected_source = ordinal_ids[int(transition["source_ordinal"])]
    expected_target = ordinal_ids[int(transition["target_ordinal"])]
    matching = [
        index
        for index, event in enumerate(events)
        if isinstance(event, Mapping)
        and event.get("relation") == "set_state"
        and event.get("source_endpoint_id") == expected_source
        and event.get("target_endpoint_id") == expected_target
    ]
    mismatches: list[str] = []
    matched_index: int | None = matching[0] if len(matching) == 1 else None
    if matched_index is None:
        mismatches.append("expected_transition_not_unique")
    initial = expected.get("initial_ordinal")
    if initial is not None and plan.get("initial_endpoint_id") != ordinal_ids[int(initial)]:
        mismatches.append("initial_endpoint")
    exact_count = expected.get("exact_event_count")
    if exact_count is not None and len(events) != int(exact_count):
        mismatches.append("exact_event_count")
    render = result.get("render")
    schedule = render.get("schedule") if isinstance(render, Mapping) else None
    if not isinstance(schedule, Mapping):
        raise EvaluateRunError(
            "verified renderer schedule disappeared"
        )
    final = expected.get("final_ordinal")
    if final is not None and schedule.get("final_endpoint_id") != ordinal_ids[int(final)]:
        mismatches.append("final_endpoint")
    route_report = {
        "ordinal_map_source": "v4-verified completed endpoint_records",
        "ordinal_ids": list(ordinal_ids),
        "caller_supplied_ordinal_map": False,
        "matched_event_index": matched_index,
        "route_constraint_pass": not mismatches,
        "mismatches": mismatches,
        "initial_endpoint_scored": initial is not None,
        "event_count_scored": exact_count is not None,
        "final_endpoint_scored": final is not None,
        "unmentioned_event_count_or_order_constraints_added": False,
    }

    failures: list[str] = []
    measurements: dict[str, Any] = {}
    if matched_index is None:
        failures.append("expected_transition_missing")
    else:
        schedule_events = schedule.get("events")
        timing = result.get("timing")
        if (
            not isinstance(schedule_events, list)
            or matched_index >= len(schedule_events)
            or not isinstance(timing, Mapping)
        ):
            raise EvaluateRunError(
                "renderer schedule no longer matches the validated transition"
            )
        scheduled = schedule_events[matched_index]
        start = int(timing["active_start_sample_relative"]) + int(
            scheduled["start_sample"]
        )
        stop = int(timing["active_start_sample_relative"]) + int(
            scheduled["stop_sample_inclusive"]
        )
        source_ordinal = int(transition["source_ordinal"])
        target_ordinal = int(transition["target_ordinal"])
        source_curve = weights[source_ordinal, start : stop + 1]
        target_curve = weights[target_ordinal, start : stop + 1]
        if source_curve.size == 0 or target_curve.size == 0:
            raise EvaluateRunError(
                "matched transition has no resolved control samples"
            )
        tolerance = 1e-10
        target_nondecreasing = bool(np.all(np.diff(target_curve) >= -tolerance))
        source_nonincreasing = bool(np.all(np.diff(source_curve) <= tolerance))
        target_positive_net_progress = bool(
            float(target_curve[-1]) > float(target_curve[0]) + tolerance
        )
        source_positive_net_progress = bool(
            float(source_curve[-1]) < float(source_curve[0]) - tolerance
        )
        positive_target_exposure = bool(
            float(np.max(target_curve)) > float(target_curve[0]) + tolerance
        )
        if not target_nondecreasing:
            failures.append("target_weight_not_nondecreasing")
        if not source_nonincreasing:
            failures.append("source_weight_not_nonincreasing")
        if not target_positive_net_progress:
            failures.append("target_weight_has_no_positive_net_progress")
        if not source_positive_net_progress:
            failures.append("source_weight_has_no_positive_net_progress")
        if not positive_target_exposure:
            failures.append("target_has_no_positive_exposure_above_event_start")
        completion = case["qualitative_curve_constraints"][0]["completion"]
        amount = float(events[matched_index]["target_amount"])
        if completion == "unspecified":
            completion_pass = True
        elif completion == "full_target":
            completion_pass = math.isclose(amount, 1.0, abs_tol=1e-12)
        else:
            completion_pass = 0.0 < amount < 1.0
        if not completion_pass:
            failures.append("qualitative_completion")
        measurements = {
            "event_start_sample": start,
            "event_stop_sample_inclusive": stop,
            "target_weight_nondecreasing": target_nondecreasing,
            "source_weight_nonincreasing": source_nonincreasing,
            "target_weight_positive_net_progress": target_positive_net_progress,
            "source_weight_positive_net_progress": source_positive_net_progress,
            "positive_target_exposure_above_event_start": positive_target_exposure,
            "target_weight_start": float(target_curve[0]),
            "target_weight_end": float(target_curve[-1]),
            "source_weight_start": float(source_curve[0]),
            "source_weight_end": float(source_curve[-1]),
            "completion_constraint": completion,
            "observed_target_amount": amount,
            "completion_constraint_pass": completion_pass,
            "numeric_amount_or_percent_target_preregistered": False,
        }
    curve_report = {
        "computed_curve_constraint_pass": not failures,
        "failures": failures,
        "measurements": measurements,
        "evaluated_property": (
            "resolved endpoint-weight direction and strictly positive net progress only"
        ),
        "gradual_duration_or_rate_appropriateness_evaluated": False,
        "linear_anchor_or_weight_audio_correlation_claimed": False,
        "arbitrary_multi_event_generalization_claimed": False,
        "numeric_progress_magnitude_threshold_invented": False,
    }
    return route_report, curve_report, matched_index


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, np.finfo(np.float64).tiny))


def _barwise_delta_proxy(
    *,
    case: Mapping[str, Any],
    run: Path,
    result: Mapping[str, Any],
    weights: np.ndarray,
    matched_event_index: int | None,
) -> dict[str, Any]:
    before, before_sr = sf.read(
        run / "target_before_FLOAT.wav", dtype="float32", always_2d=True
    )
    after, after_sr = sf.read(
        run / "target_after_FLOAT.wav", dtype="float32", always_2d=True
    )
    if (
        before_sr != manifest.SAMPLE_RATE
        or after_sr != before_sr
        or before.shape != after.shape
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
    ):
        raise EvaluateRunError(
            "completed before/after audio shape, rate, or finiteness changed"
        )
    difference = np.ascontiguousarray(after - before, dtype=np.float64)
    activation = np.load(run / "boundary_activation.npy", allow_pickle=False)
    if activation.shape != (before.shape[0],) or not np.isfinite(activation).all():
        raise EvaluateRunError(
            "boundary activation shape or finiteness changed"
        )
    active_indexes = np.flatnonzero(activation > 0.0)
    first_active = int(active_indexes[0]) if active_indexes.size else before.shape[0]
    pre_delta = difference[:first_active]
    preactivation_exact = bool(
        np.array_equal(pre_delta, np.zeros_like(pre_delta))
    )
    causal_null_pass: bool | None = (
        preactivation_exact if pre_delta.shape[0] > 0 else None
    )

    context = _read_json(run / "automation" / "llm_safe_context.json")
    structures = context.get("AVAILABLE_MUSIC_STRUCTURE", {}).get(
        "section_bar_boundaries"
    )
    if not isinstance(structures, list) or len(structures) != 1:
        raise EvaluateRunError(
            "barwise evaluator requires one verified selected-section grid"
        )
    structure = structures[0]
    if not isinstance(structure, Mapping) or not isinstance(structure.get("bars"), list):
        raise EvaluateRunError(
            "barwise structure shape changed"
        )
    timing = result["timing"]
    section_start = float(structure["section_start_sec"])
    active_start = int(timing["active_start_sample_relative"])
    active_stop = int(timing["active_stop_sample_relative"])
    event_start, event_stop = active_start, active_stop
    if matched_event_index is not None:
        event = result["render"]["schedule"]["events"][matched_event_index]
        event_start = active_start + int(event["start_sample"])
        event_stop = active_start + int(event["stop_sample_inclusive"]) + 1
    source_ordinal = int(case["expected_route"]["transition"]["source_ordinal"])
    target_ordinal = int(case["expected_route"]["transition"]["target_ordinal"])
    config = case["metric_config"]
    rows: list[dict[str, Any]] = []
    total_input_energy = 0.0
    total_delta_energy = 0.0
    active_bar_count = 0
    for bar in structure["bars"]:
        if not isinstance(bar, Mapping):
            raise EvaluateRunError(
                "barwise structure entry changed"
            )
        raw_start = active_start + int(
            round((float(bar["start_sec"]) - section_start) * before_sr)
        )
        raw_stop = active_start + int(
            round((float(bar["end_sec"]) - section_start) * before_sr)
        )
        start = max(active_start, event_start, raw_start)
        stop = min(active_stop, event_stop, raw_stop)
        if stop <= start:
            continue
        source_audio = np.asarray(before[start:stop], dtype=np.float64)
        delta_audio = difference[start:stop]
        input_energy = float(np.square(source_audio).sum())
        delta_energy = float(np.square(delta_audio).sum())
        input_rms = math.sqrt(input_energy / max(source_audio.size, 1))
        relative = math.sqrt(
            delta_energy / max(input_energy, np.finfo(np.float64).tiny)
        )
        input_db = _db(input_rms)
        active = input_db >= float(config["activity_rms_dbfs_threshold"])
        if active:
            active_bar_count += 1
            total_input_energy += input_energy
            total_delta_energy += delta_energy
        source_weight = float(np.mean(weights[source_ordinal, start:stop]))
        target_weight = float(np.mean(weights[target_ordinal, start:stop]))
        rows.append(
            {
                "bar_index": int(bar["bar_index"]),
                "start_sample": start,
                "stop_sample_exclusive": stop,
                "input_rms_dbfs": input_db,
                "target_stem_activity_by_frozen_energy_threshold": active,
                "d_relative_db": _db(relative),
                "source_endpoint_weight_mean": source_weight,
                "target_endpoint_weight_mean": target_weight,
                "event_axis_target_minus_source_weight_mean": (
                    target_weight - source_weight
                ),
            }
        )
    weighted_db: float | None = None
    if total_input_energy > 0.0:
        weighted_db = _db(math.sqrt(total_delta_energy / total_input_energy))
    enough_bars = active_bar_count >= int(config["minimum_active_bars"])
    threshold_pass = bool(
        enough_bars
        and weighted_db is not None
        and weighted_db >= float(config["delta_presence_relative_db_threshold"])
    )
    return {
        "metric": "target-stem-activity/energy-weighted d(t)",
        "scope": config["metric_scope"],
        "measurement_interval": "matched transition intersected with selected-section bars",
        "activity_detector": "frozen input RMS threshold only; not a vocal classifier",
        "bars": rows,
        "active_bar_count": active_bar_count,
        "minimum_active_bars": int(config["minimum_active_bars"]),
        "activity_rms_dbfs_threshold": float(config["activity_rms_dbfs_threshold"]),
        "energy_weighted_d_relative_db": weighted_db,
        "delta_presence_relative_db_threshold": float(
            config["delta_presence_relative_db_threshold"]
        ),
        "delta_presence_proxy_pass": threshold_pass,
        "threshold_freeze_scope": config["threshold_freeze_scope"],
        "designed_after_prior_results": config["designed_after_prior_results"],
        "independent_confirmation_claimed": config[
            "independent_confirmation_claimed"
        ],
        "human_audibility_claimed": False,
        "style_transition_audibility_claimed": False,
        "pre_activation_sample_count": int(pre_delta.shape[0]),
        "causal_pre_activation_delta_is_bit_exact_zero": causal_null_pass,
        "pre_activation_delta_peak_abs": float(
            np.max(np.abs(pre_delta), initial=0.0)
        ),
        "full_render_history_used": True,
        "bars_not_rendered_independently": True,
        "reverb_history_preserved": True,
        "plan_axis_is_descriptive_not_assumed_linear": True,
    }


def _integrity_and_safety(
    *, run: Path, result: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    after, sample_rate = sf.read(
        run / "target_after_FLOAT.wav", dtype="float32", always_2d=True
    )
    finite = bool(np.isfinite(after).all())
    peak = float(np.max(np.abs(after), initial=0.0))
    clipping = int(np.count_nonzero(np.abs(after) > 1.0))
    diagnostics = result.get("render", {}).get("diagnostics", {})
    reverb = diagnostics.get("reverb") if isinstance(diagnostics, Mapping) else None
    tail_safe = bool(
        not isinstance(reverb, Mapping)
        or not reverb.get("tail_expected")
        or (
            reverb.get("tail_context_complete") is True
            and reverb.get("post_send_residual_matches_wet_tail") is True
            and reverb.get("tail_preserved") is True
        )
    )
    verifier_pass = verification.get("verified") is True
    return {
        "layer": "execution_integrity_and_safety_only",
        "v4_source_specific_completed_verifier_pass": verifier_pass,
        "artifact_checks": deepcopy(dict(verification.get("artifact_checks", {}))),
        "sample_rate_hz": int(sample_rate),
        "output_finite": finite,
        "output_peak_abs": peak,
        "output_samples_over_1_0": clipping,
        "reverb_tail_safe": tail_safe,
        "integrity_safety_pass": bool(
            verifier_pass and finite and clipping == 0 and tail_safe
        ),
        "semantic_judgments_used": False,
    }


def evaluate_completed_case(
    *, semantic_prepared: Path, completed_run: Path
) -> dict[str, Any]:
    semantic_path = semantic_prepared.resolve()
    run = completed_run.resolve()
    case, case_digest, _ = semantic_prereg.verify_prepared_case(semantic_path)
    run_prereg, result, verification, source_binding = _verify_source_and_hash_chain(
        semantic_path=semantic_path,
        semantic_case=case,
        semantic_digest=case_digest,
        run=run,
    )
    budget = run_prereg.get("exact_work_budget")
    if not isinstance(budget, Mapping):
        raise EvaluateRunError(
            "frozen execution work budget disappeared"
        )
    blinding = _verify_request_blinding(
        case=case,
        run=run,
        expected_calls=int(budget["total_api_calls"]),
    )
    plan_source = source_binding[
        "plan_source_derived_from_verified_execution_artifact"
    ]
    native_provenance = _native_response_provenance(
        case=case,
        run=run,
        run_prereg=run_prereg,
        plan_source=plan_source,
        expected_calls=int(budget["total_api_calls"]),
    )
    ordinal_ids = _ordinal_ids_from_verified_result(case, result)
    weights = np.load(run / "resolved_endpoint_weights.npy", allow_pickle=False)
    before, _ = sf.read(
        run / "target_before_FLOAT.wav", dtype="float32", always_2d=True
    )
    if (
        weights.ndim != 2
        or weights.shape != (len(ordinal_ids), before.shape[0])
        or not np.isfinite(weights).all()
    ):
        raise EvaluateRunError(
            "verified endpoint-weight shape or finiteness changed"
        )
    route, curve, matched_index = _evaluate_route_and_curve(
        case=case,
        result=result,
        ordinal_ids=ordinal_ids,
        weights=weights,
    )
    delta = _barwise_delta_proxy(
        case=case,
        run=run,
        result=result,
        weights=weights,
        matched_event_index=matched_index,
    )
    artifact_checks = verification.get("artifact_checks")
    exact_replay = bool(
        isinstance(artifact_checks, Mapping)
        and artifact_checks.get("renderer_replay") is True
        and artifact_checks.get("control_hashes") is True
        and artifact_checks.get("decoded_audio_hashes") is True
    )
    causal_null = delta["causal_pre_activation_delta_is_bit_exact_zero"]
    plan_audio = {
        "primary_metric": "v4 exact full renderer/control/decoded-audio replay",
        "exact_full_replay_pass": exact_replay,
        "causal_pre_activation_null_pass": causal_null,
        "causal_null_test_available": causal_null is not None,
        "full_history_reverb_preserved": True,
        "plan_audio_self_consistency_pass": bool(
            exact_replay and causal_null is not False
        ),
        "barwise_delta_is_secondary_descriptive_proxy": True,
        "style_realization_inferred_from_signal_delta": False,
    }
    integrity = _integrity_and_safety(
        run=run, result=result, verification=verification
    )

    paraphrase = (
        case["evidence"]["final_evidence_class"]
        == "lexically_disjoint_semantic_paraphrase"
    )
    annotation = case["annotation_protocol"]
    manual_direct_cue_review_ok = bool(
        annotation["direct_cue_review_completed_before_execution"] is True
        and annotation["direct_cue_review_conclusion"]
        == "no_unlisted_direct_selector_cue_found"
    )
    selection_eligible = bool(
        plan_source == execute_run.LIVE_PLAN_SOURCE
        and native_provenance["native_response_metadata_complete"]
        and paraphrase
        and manual_direct_cue_review_ok
    )
    temporal_eligible = bool(
        plan_source == execute_run.LIVE_PLAN_SOURCE
        and native_provenance["native_response_metadata_complete"]
    )
    semantic_planning_pass = bool(
        route["route_constraint_pass"]
        and curve["computed_curve_constraint_pass"]
    )
    smoke_reasons: list[str] = []
    if plan_source != execute_run.LIVE_PLAN_SOURCE:
        smoke_reasons.append("test_double_plan_source")
    if not native_provenance["native_response_metadata_complete"]:
        smoke_reasons.append("native_response_metadata_incomplete")
    if not paraphrase:
        smoke_reasons.append("overlap_direct_cue_or_selection_design_blocker")
    if paraphrase and not manual_direct_cue_review_ok:
        smoke_reasons.append("manual_direct_cue_review_missing_or_not_clear")

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "case_id": case["case_id"],
        "semantic_pre_registration_sha256": case_digest,
        "preexecution_binding_and_blinding": {
            **source_binding,
            **blinding,
            "native_response_provenance": native_provenance,
        },
        "semantic_routing": {
            "automatic_evidence_class": case["evidence"][
                "automatic_evidence_class"
            ],
            "manual_review_evidence_class": case["evidence"][
                "manual_review_evidence_class"
            ],
            "final_evidence_class": case["evidence"]["final_evidence_class"],
            "manual_review_downgraded_automatic_paraphrase": case["evidence"][
                "manual_review_downgraded_automatic_paraphrase"
            ],
            "selection_design_blockers": list(
                case["evidence"]["selection_design_blockers"]
            ),
            "automatic_cue_detector_exhaustive": case["evidence"][
                "automatic_cue_detector_exhaustive"
            ],
            "manual_direct_cue_review_completed_before_execution": annotation[
                "direct_cue_review_completed_before_execution"
            ],
            "manual_direct_cue_review_conclusion": annotation[
                "direct_cue_review_conclusion"
            ],
            "manual_direct_cue_review_bound_to_instruction_sha256": annotation[
                "reviewed_automation_instruction_utf8_sha256"
            ],
            "route": route,
            "case_level_selection_evidence_eligible": selection_eligible,
            "case_level_selection_evidence_pass": (
                semantic_planning_pass if selection_eligible else None
            ),
            "smoke_only_route_pass": (
                route["route_constraint_pass"] if not selection_eligible else None
            ),
            "smoke_only_semantic_planning_pass": (
                semantic_planning_pass if not selection_eligible else None
            ),
            "smoke_only_reasons": smoke_reasons,
            "general_selection_capability_claim_allowed": False,
        },
        "qualitative_curve_compliance": {
            **curve,
            "case_level_curve_direction_evidence_eligible": temporal_eligible,
            "case_level_curve_direction_evidence_pass": (
                curve["computed_curve_constraint_pass"]
                if temporal_eligible
                else None
            ),
            "test_double_curve_direction_pipeline_smoke_pass": (
                curve["computed_curve_constraint_pass"]
                if not temporal_eligible
                else None
            ),
            "evidence_scope": (
                "live_native_metadata_complete_case_level_curve_direction_evidence"
                if temporal_eligible
                else "pipeline_smoke_only_not_model_capability_evidence"
            ),
        },
        "plan_audio_causal_self_consistency": plan_audio,
        "target_stem_activity_energy_weighted_delta": delta,
        "execution_integrity_and_safety": integrity,
        "endpoint_style_realization_claimed": False,
        "human_listening_success_claimed": False,
        "independent_confirmation_claimed": False,
        "cross_layer_combination_forbidden": True,
        "combined_overall_pass": None,
        "evaluator_api_calls": 0,
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_evaluation(
    *, semantic_prepared: Path, completed_run: Path, output: Path
) -> Path:
    semantic_prepared = semantic_prepared.resolve()
    completed_run = completed_run.resolve()
    output = output.resolve()
    if output.exists():
        raise EvaluateRunError(
            f"refusing to overwrite evaluation: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging_", dir=output.parent))
    try:
        report = evaluate_completed_case(
            semantic_prepared=semantic_prepared, completed_run=completed_run
        )
        artifact = {
            "evaluator_version": EVALUATOR_VERSION,
            "status": "semantic_evaluation_completed",
            "evaluator_code": _file_identity(Path(__file__)),
            "sources": {
                "semantic_prepared_directory": str(semantic_prepared),
                "completed_run_directory": str(completed_run),
            },
            "report": report,
        }
        result_path = staging / "EVALUATION_RESULT.json"
        _write_json_exclusive(result_path, artifact)
        digest = manifest.file_sha256(result_path)
        (staging / "EVALUATION_RESULT.sha256").write_text(
            f"{digest}  EVALUATION_RESULT.json\n", encoding="ascii"
        )
        _write_json_exclusive(
            staging / "COMPLETED.json",
            {
                "status": "semantic_evaluation_completed",
                "evaluator_version": EVALUATOR_VERSION,
                "evaluator_code": _file_identity(Path(__file__)),
                "evaluation_result_sha256": digest,
                "semantic_pre_registration_sha256": report[
                    "semantic_pre_registration_sha256"
                ],
                "execution_pre_registration_sha256": report[
                    "preexecution_binding_and_blinding"
                ]["execution_pre_registration_sha256"],
                "plan_source": report["preexecution_binding_and_blinding"][
                    "plan_source_derived_from_verified_execution_artifact"
                ],
                "api_calls": 0,
                "optimizer_runs": 0,
                "v4_deterministic_renderer_replay_used": True,
                "atomic_persistence": True,
            },
        )
    except Exception as error:
        for incompatible in (
            staging / "EVALUATION_RESULT.json",
            staging / "EVALUATION_RESULT.sha256",
            staging / "COMPLETED.json",
        ):
            incompatible.unlink(missing_ok=True)
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "status": "semantic_evaluation_failed",
                "evaluator_version": EVALUATOR_VERSION,
                "evaluator_code": _file_identity(Path(__file__)),
                "semantic_prepared_directory": str(semantic_prepared),
                "completed_run_directory": str(completed_run),
                "error_type": type(error).__name__,
                "error": str(error),
                "api_calls": 0,
                "optimizer_runs": 0,
                "traceback": traceback.format_exc(),
            },
        )
        os.replace(staging, output)
        raise
    os.replace(staging, output)
    return output


def verify_saved_evaluation(path: Path) -> dict[str, Any]:
    path = path.resolve()
    allowed = {
        "EVALUATION_RESULT.json",
        "EVALUATION_RESULT.sha256",
        "COMPLETED.json",
    }
    if (
        not path.is_dir()
        or (path / "FAILURE.json").exists()
        or {item.name for item in path.iterdir()} != allowed
    ):
        raise EvaluateRunError(
            "saved semantic evaluation file set changed or contains FAILURE"
        )
    result_path = path / "EVALUATION_RESULT.json"
    digest = manifest.file_sha256(result_path)
    if path.joinpath("EVALUATION_RESULT.sha256").read_text(
        encoding="ascii"
    ).split() != [digest, "EVALUATION_RESULT.json"]:
        raise EvaluateRunError(
            "saved semantic evaluation sidecar changed"
        )
    artifact = _read_json(result_path)
    completed = _read_json(path / "COMPLETED.json")
    current_code = _file_identity(Path(__file__))
    if artifact.get("evaluator_code") != current_code:
        raise EvaluateRunError(
            "saved semantic evaluator code identity changed"
        )
    sources = artifact.get("sources")
    if not isinstance(sources, Mapping):
        raise EvaluateRunError(
            "saved semantic evaluation source binding disappeared"
        )
    recomputed = evaluate_completed_case(
        semantic_prepared=Path(str(sources["semantic_prepared_directory"])),
        completed_run=Path(str(sources["completed_run_directory"])),
    )
    expected_artifact = {
        "evaluator_version": EVALUATOR_VERSION,
        "status": "semantic_evaluation_completed",
        "evaluator_code": current_code,
        "sources": dict(sources),
        "report": recomputed,
    }
    expected_completed = {
        "status": "semantic_evaluation_completed",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_code": current_code,
        "evaluation_result_sha256": digest,
        "semantic_pre_registration_sha256": recomputed[
            "semantic_pre_registration_sha256"
        ],
        "execution_pre_registration_sha256": recomputed[
            "preexecution_binding_and_blinding"
        ]["execution_pre_registration_sha256"],
        "plan_source": recomputed["preexecution_binding_and_blinding"][
            "plan_source_derived_from_verified_execution_artifact"
        ],
        "api_calls": 0,
        "optimizer_runs": 0,
        "v4_deterministic_renderer_replay_used": True,
        "atomic_persistence": True,
    }
    if artifact != expected_artifact or completed != expected_completed:
        raise EvaluateRunError(
            "saved semantic evaluation result or completion marker changed"
        )
    return {
        "verified": True,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_result_sha256": digest,
        "semantic_pre_registration_sha256": recomputed[
            "semantic_pre_registration_sha256"
        ],
        "plan_source": expected_completed["plan_source"],
        "api_calls": 0,
    }
