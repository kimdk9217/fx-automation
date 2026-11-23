"""LLM request and validation for endpoints described in text."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import numpy as np

import endpoint_manifest as manifest
import llm
import reference_bridge as bridge
import reference_matcher as matcher

CONTRACT_VERSION = "dynamic-static-endpoint-contract-v3"
STATIC_PLAN_VERSION = "dynamic-static-endpoint-plan-v3"
EQ_BANDS = ("low_shelf", "band0", "band1", "band2", "band3", "high_shelf")
REVERB_BANDS = 12
REVERB_BAND_ORDER_DESCRIPTION = (
    "backend order: low-pass 12 Hz, octave centers 31.5, 63, 125, 250, "
    "500, 1000, 2000, 4000, 8000, 16000 Hz, then high-pass 18000 Hz"
)
SYSTEM_PROMPT = """\
Convert the caller's exact natural-language sound description into one static
audio-effect endpoint. The target stem and processor mode are locked by the
caller. Return only generic DSP values allowed by the supplied JSON schema.

This request creates no timeline. Do not output events, durations, automation
amounts, initial or final states, endpoint order, presets, candidate lists,
style dictionaries, lookup tables, examples, retries, or repairs. Do not add a
second interpretation of the caller text. Return one schema-valid JSON object.
"""


class DynamicEndpointContractError(ValueError):
    pass


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise DynamicEndpointContractError(f"locked backend artifact is missing: {path}")
    return {"path": str(path), "sha256": manifest.file_sha256(path)}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_schema(low: float, high: float) -> dict[str, Any]:
    return {"type": "number", "minimum": float(low), "maximum": float(high)}


def _eq_schema() -> dict[str, Any]:
    ranges = matcher.official_eq_parameter_ranges()
    properties: dict[str, Any] = {}
    for band in EQ_BANDS:
        gain = ranges[f"{band}_gain_db"]
        frequency = ranges[f"{band}_cutoff_freq"]
        q_factor = ranges[f"{band}_q_factor"]
        properties[band] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["gain_db", "frequency_hz", "q_factor"],
            "properties": {
                "gain_db": _number_schema(*gain),
                "frequency_hz": _number_schema(*frequency),
                "q_factor": _number_schema(*q_factor),
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(EQ_BANDS),
        "properties": properties,
    }


def _reverb_schema() -> dict[str, Any]:
    array = {
        "type": "array",
        "minItems": REVERB_BANDS,
        "maxItems": REVERB_BANDS,
        "items": _number_schema(0.0, 1.0),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["band_gains", "band_decays", "mix"],
        "properties": {
            "band_gains": {
                **deepcopy(array),
                "description": (
                    "Twelve linear noise-band coefficients in "
                    + REVERB_BAND_ORDER_DESCRIPTION
                    + "; zero suppresses a band and one keeps its full generated level."
                ),
            },
            "band_decays": {
                **deepcopy(array),
                "description": (
                    "Twelve normalized decay-rate controls in "
                    + REVERB_BAND_ORDER_DESCRIPTION
                    + "; larger values produce a faster envelope decay. Values are "
                    "not seconds or RT60."
                ),
            },
            "mix": {
                **_number_schema(0.0, 1.0),
                "description": (
                    "Static endpoint dry/wet crossfade coefficient; zero is dry and "
                    "one is the full generated wet signal."
                ),
            },
        },
    }


def make_static_schema(*, effect_mode: str, target_stem: str) -> dict[str, Any]:
    if effect_mode not in manifest.EFFECT_MODES:
        raise DynamicEndpointContractError("unsupported explicit effect mode")
    if target_stem not in manifest.TARGET_STEMS:
        raise DynamicEndpointContractError("unsupported target stem")
    parameter_properties: dict[str, Any] = {}
    required: list[str] = []
    if effect_mode in {"eq", "eq_reverb"}:
        parameter_properties["eq"] = _eq_schema()
        required.append("eq")
    if effect_mode in {"reverb", "eq_reverb"}:
        parameter_properties["reverb"] = _reverb_schema()
        required.append("reverb")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DynamicStaticEndpointPlanV3",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "target_stem", "effect_mode", "parameters"],
        "properties": {
            "schema_version": {"const": STATIC_PLAN_VERSION},
            "target_stem": {"const": target_stem},
            "effect_mode": {"const": effect_mode},
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": parameter_properties,
            },
        },
    }


def build_static_request(
    endpoint: manifest.CanonicalEndpoint,
    *,
    target_stem: str,
    model: str,
    reasoning_effort: str | None,
    base_url: str,
) -> dict[str, Any]:
    if endpoint.kind != "natural_language" or endpoint.user_description is None:
        raise DynamicEndpointContractError("static text request got another endpoint kind")
    schema = make_static_schema(
        effect_mode=endpoint.effect_mode, target_stem=target_stem
    )
    context = {
        "USER_SOUND_DESCRIPTION_EXACT": endpoint.user_description,
        "LOCKED_TARGET_STEM": target_stem,
        "LOCKED_EFFECT_MODE": endpoint.effect_mode,
        "AUTHORITY": {
            "endpoint_is_static": True,
            "timeline_and_automation_owned_by_later_planner": True,
            "caller_text_rewritten": False,
        },
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"dynamic_static_{endpoint.effect_mode}_endpoint_v3",
                "strict": True,
                "schema": schema,
            },
        },
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": payload["messages"],
        "response_format": llm._strict_response_format(
            payload["response_format"]
        ),
    }
    if llm.is_reasoning_model(model):
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
    else:
        body["temperature"] = 0.0
    native_request = {
        "endpoint": f"{base_url.rstrip('/')}/chat/completions",
        "body": body,
        "api_key_recorded": False,
    }
    return {
        "schema": schema,
        "schema_sha256": _sha_json(schema),
        "context": context,
        "context_sha256": _sha_json(context),
        "payload": payload,
        "payload_sha256": _sha_json(payload),
        "native_request": native_request,
        "native_request_sha256": _sha_json(native_request),
    }


def validate_static_plan(
    raw: Any, *, endpoint: manifest.CanonicalEndpoint, target_stem: str
) -> dict[str, Any]:
    schema = make_static_schema(
        effect_mode=endpoint.effect_mode, target_stem=target_stem
    )
    try:
        jsonschema.Draft202012Validator(schema).validate(raw)
    except jsonschema.ValidationError as error:
        raise DynamicEndpointContractError(
            f"static endpoint plan failed schema validation: {error.message}"
        ) from error
    return deepcopy(dict(raw))


def normalized_parameters_from_plan(
    raw: Any,
    *,
    endpoint: manifest.CanonicalEndpoint,
    target_stem: str,
    renderer_mode: str | None = None,
) -> np.ndarray:
    plan = validate_static_plan(raw, endpoint=endpoint, target_stem=target_stem)
    values: list[float] = []
    parameters = plan["parameters"]
    if endpoint.effect_mode in {"eq", "eq_reverb"}:
        ranges = matcher.official_eq_parameter_ranges()
        for band in EQ_BANDS:
            row = parameters["eq"][band]
            for field, suffix in (
                ("gain_db", "gain_db"),
                ("frequency_hz", "cutoff_freq"),
                ("q_factor", "q_factor"),
            ):
                low, high = ranges[f"{band}_{suffix}"]
                value = float(row[field])
                normalized = (value - low) / (high - low)
                if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                    raise DynamicEndpointContractError("EQ normalization left [0,1]")
                values.append(float(normalized))
    if endpoint.effect_mode in {"reverb", "eq_reverb"}:
        reverb = parameters["reverb"]
        values.extend(float(value) for value in reverb["band_gains"])
        values.extend(float(value) for value in reverb["band_decays"])
        values.append(float(reverb["mix"]))
    effective_renderer_mode = endpoint.effect_mode if renderer_mode is None else renderer_mode
    if effective_renderer_mode not in manifest.EFFECT_MODES:
        raise DynamicEndpointContractError("renderer mode is invalid")
    if effective_renderer_mode != endpoint.effect_mode:
        if effective_renderer_mode != "eq_reverb":
            raise DynamicEndpointContractError(
                "single-effect renderer cannot contain another endpoint mode"
            )
        if endpoint.effect_mode == "eq":
            values.extend([0.5] * 24)
            values.append(0.0)
        elif endpoint.effect_mode == "reverb":
            raise DynamicEndpointContractError(
                "reverb-only endpoint in an eq_reverb renderer requires a "
                "reviewed per-effect applicability mask"
            )
        else:
            raise DynamicEndpointContractError("unsupported endpoint embedding")
    result = np.asarray([values], dtype=np.float32)
    expected = bridge.parameter_count(effective_renderer_mode)
    if result.shape != (1, expected) or not np.isfinite(result).all():
        raise DynamicEndpointContractError("materialized parameter vector is invalid")
    return result


def renderer_embedding_contract(
    *, endpoint_mode: str, renderer_mode: str
) -> dict[str, Any]:
    if endpoint_mode not in manifest.EFFECT_MODES or renderer_mode not in manifest.EFFECT_MODES:
        raise DynamicEndpointContractError("endpoint or renderer mode is invalid")
    if endpoint_mode == renderer_mode:
        policy = "no_padding"
    elif renderer_mode == "eq_reverb" and endpoint_mode == "eq":
        policy = "append_reverb_shape_midpoints_and_exact_mix_zero_bypass"
        execution_compatible = True
        requires_effect_applicability_mask = False
    elif renderer_mode == "eq_reverb" and endpoint_mode == "reverb":
        policy = "blocked_until_exact_eq_bypass_mask_is_implemented"
        execution_compatible = False
        requires_effect_applicability_mask = True
    else:
        raise DynamicEndpointContractError(
            "endpoint mode cannot be embedded into selected renderer mode"
        )
    if endpoint_mode == renderer_mode:
        execution_compatible = True
        requires_effect_applicability_mask = False
    return {
        "endpoint_mode": endpoint_mode,
        "renderer_mode": renderer_mode,
        "endpoint_parameter_count": bridge.parameter_count(endpoint_mode),
        "renderer_parameter_count": bridge.parameter_count(renderer_mode),
        "policy": policy,
        "structural_bypass_only": (
            renderer_mode == "eq_reverb" and endpoint_mode == "eq"
        ),
        "execution_compatible": execution_compatible,
        "requires_effect_applicability_mask": requires_effect_applicability_mask,
        "style_values_injected": False,
    }


def _cache_key(payload: Mapping[str, Any]) -> str:
    return _sha_json(payload)


def natural_language_route(
    endpoint: manifest.CanonicalEndpoint,
    *,
    target: Mapping[str, Any],
    renderer_mode: str,
    model: str,
    reasoning_effort: str | None,
    base_url: str,
) -> dict[str, Any]:
    request = build_static_request(
        endpoint,
        target_stem=str(target["stem"]),
        model=model,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
    )
    cache_inputs = {
        "user_text_utf8_sha256": _sha_text(str(endpoint.user_description)),
        "target_audio_sha256": target["raw_stem"]["sha256"],
        "target_stem": target["stem"],
        "target_structure_sha256": target["structure"]["sha256"],
        "target_selected_section": target["structure"]["selected_section"],
        "target_selected_section_audio": target["selected_section_audio"],
        "section": target["section"],
        "occurrence": target["occurrence"],
        "effect_mode": endpoint.effect_mode,
        "renderer_mode": renderer_mode,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "base_url": base_url,
        "system_prompt_sha256": _sha_text(SYSTEM_PROMPT),
        "schema_sha256": request["schema_sha256"],
        "native_request_sha256": request["native_request_sha256"],
        "contract_code": _identity(Path(__file__)),
        "openai_request_code": _identity(Path(llm.__file__)),
        "matcher_code": _identity(Path(matcher.__file__)),
        "renderer_code": _identity(Path(bridge.__file__)),
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": endpoint.kind,
        "endpoint_id": endpoint.endpoint_id,
        "identifier_visibility": "hidden_internal_identifier",
        "effect_mode": endpoint.effect_mode,
        "renderer_embedding": renderer_embedding_contract(
            endpoint_mode=endpoint.effect_mode, renderer_mode=renderer_mode
        ),
        "user_description_exact": endpoint.user_description,
        "user_description_utf8_sha256": _sha_text(str(endpoint.user_description)),
        "request": request,
        "cache_key_inputs": cache_inputs,
        "cache_key_sha256": _cache_key(cache_inputs),
        "structured_api_calls_if_executed": 1,
        "optimizer_steps_if_executed": 0,
        "style_lookup_table": False,
        "style_preset": False,
        "production_examples": False,
        "injected_scenario_or_curve": False,
        "general_dsp_schema_and_ranges_only": True,
        "static_plan_parser_implemented": True,
        "materializer_executed_by_this_contract_module": False,
        "execution_allowed_by_this_contract": False,
    }


def processed_reference_route(
    endpoint: manifest.CanonicalEndpoint,
    *,
    target: Mapping[str, Any],
    renderer_mode: str,
) -> dict[str, Any]:
    if endpoint.kind != "processed_reference" or endpoint.source_audio is None:
        raise DynamicEndpointContractError("processed-reference route got another kind")
    cache_inputs = {
        "target_audio_sha256": target["raw_stem"]["sha256"],
        "reference_audio_sha256": endpoint.source_audio["sha256"],
        "target_stem": target["stem"],
        "target_structure_sha256": target["structure"]["sha256"],
        "target_selected_section": target["structure"]["selected_section"],
        "target_selected_section_audio": target["selected_section_audio"],
        "section": target["section"],
        "occurrence": target["occurrence"],
        "effect_mode": endpoint.effect_mode,
        "renderer_mode": renderer_mode,
        "matcher_code": _identity(Path(matcher.__file__)),
        "renderer_code": _identity(Path(bridge.__file__)),
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": endpoint.kind,
        "endpoint_id": endpoint.endpoint_id,
        "identifier_visibility": "hidden_internal_identifier",
        "effect_mode": endpoint.effect_mode,
        "renderer_embedding": renderer_embedding_contract(
            endpoint_mode=endpoint.effect_mode, renderer_mode=renderer_mode
        ),
        "source_audio": dict(endpoint.source_audio),
        "current_target_audio": dict(target["raw_stem"]),
        "matcher_backend": _identity(Path(matcher.__file__)),
        "renderer_backend": _identity(Path(bridge.__file__)),
        "cache_key_inputs": cache_inputs,
        "cache_key_sha256": _cache_key(cache_inputs),
        "structured_api_calls_if_executed": 0,
        "optimizer_runs_if_executed": 1,
        "optimizer_steps_per_run_if_executed": matcher.MATCH_STEPS,
        "effect_mode_is_caller_declared": True,
        "effect_mode_inference": False,
        "reference_dry_required": False,
        "style_lookup_table": False,
        "style_preset": False,
        "injected_scenario_or_curve": False,
        "materializer_executed_by_this_contract_module": False,
        "execution_allowed_by_this_contract": False,
        "required_before_live_execution": [
            "frozen Fx-Encoder++ checkpoint hash and official checkout identity",
            "current target section slice hash and deterministic resampling policy",
            "complete imported renderer and encoder code closure",
            "reviewed optimizer safety and completion gates",
        ],
    }


def build_future_route(
    endpoint: manifest.CanonicalEndpoint,
    *,
    target: Mapping[str, Any],
    renderer_mode: str,
    model: str,
    reasoning_effort: str | None,
    base_url: str,
) -> dict[str, Any]:
    if endpoint.kind == "natural_language":
        return natural_language_route(
            endpoint,
            target=target,
            renderer_mode=renderer_mode,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
        )
    if endpoint.kind == "processed_reference":
        return processed_reference_route(
            endpoint, target=target, renderer_mode=renderer_mode
        )
    raise DynamicEndpointContractError(f"unsupported endpoint kind {endpoint.kind!r}")


def route_budget(routes: list[Mapping[str, Any]]) -> dict[str, int]:
    endpoint_calls = sum(
        int(route.get("structured_api_calls_if_executed", 0)) for route in routes
    )
    optimizer_runs = sum(
        int(route.get("optimizer_runs_if_executed", 0)) for route in routes
    )
    optimizer_steps = sum(
        int(route.get("optimizer_runs_if_executed", 0))
        * int(route.get("optimizer_steps_per_run_if_executed", 0))
        for route in routes
    )
    return {
        "endpoint_structured_api_calls_if_reviewed_live_executor_runs": endpoint_calls,
        "automation_planner_api_calls_if_reviewed_live_executor_runs": 1,
        "total_structured_api_calls_if_reviewed_live_executor_runs": endpoint_calls + 1,
        "reference_optimizer_runs_if_reviewed_live_executor_runs": optimizer_runs,
        "reference_optimizer_steps_if_reviewed_live_executor_runs": optimizer_steps,
    }
