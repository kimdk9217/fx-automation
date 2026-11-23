"""Automation plan schema, validation and conversion of events into endpoint weights."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import jsonschema
import numpy as np
import torch

import reference_bridge as bridge

SCHEMA_VERSION = "multi-endpoint-automation-plan-v1"
MAX_EVENTS = 12
ENDPOINT_ROLES = ("dry", "natural_language_style", "processed_reference")
EVENT_FIELDS = (
    "id",
    "instruction_evidence",
    "relation",
    "source_endpoint_id",
    "target_endpoint_id",
    "target_amount",
    "duration_weight",
    "handle1_time_fraction",
    "handle2_time_fraction",
    "handle1_value_fraction",
    "handle2_value_fraction",
)
_ENDPOINT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
_ENDPOINT_NAME = re.compile(r"^[^\W_]+(?:[ _-][^\W_]+){0,7}$", re.UNICODE)
_FORBIDDEN_ENDPOINT_NAME_TOKENS = (
    "assistant",
    "audio",
    "clap",
    "coefficient",
    "cutoff",
    "embedding",
    "gain",
    "ignore",
    "instruction",
    "json",
    "mfcc",
    "parameter",
    "prompt",
    "q_factor",
    "system",
)
_RUNTIME_ONLY_KEYS = {
    "audio",
    "audio_path",
    "audio_samples",
    "embedding",
    "eq_parameters",
    "mfcc",
    "normalized_parameters",
    "parameter_vector",
    "path",
    "physical_parameters",
    "raw_audio",
    "reverb_parameters",
    "waveform",
}
_RUNTIME_ONLY_KEY_FRAGMENTS = (
    "audio_data",
    "audio_url",
    "clap",
    "coefficients",
    "dsp_coeff",
    "embedding",
    "filter_gain",
    "mfcc",
    "normalized_parameter",
    "parameter_vector",
    "physical_parameter",
    "raw_pcm",
    "waveform",
)

def _value_unit_schema(unit: str, **value_constraints: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "unit"],
        "properties": {
            "value": {"type": "number", **value_constraints},
            "unit": {"const": unit},
        },
    }


_BAND_SUMMARY_SCHEMAS = []
for _band, _low, _high in (
    ("sub_bass", 0.0, 80.0),
    ("bass", 80.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("high", 6000.0, None),
):
    _BAND_SUMMARY_SCHEMAS.append(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["band", "low_hz", "high_hz", "energy_share_percent"],
            "properties": {
                "band": {"const": _band},
                "low_hz": {"const": _low},
                "high_hz": (
                    {"const": _high}
                    if _high is not None
                    else {"type": "number", "exclusiveMinimum": 6000.0}
                ),
                "energy_share_percent": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
            },
        }
    )
DSP_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_silent", "level", "spectrum", "stereo", "analysis_counts"],
    "properties": {
        "is_silent": {"type": "boolean"},
        "level": {
            "type": "object",
            "additionalProperties": False,
            "required": ["rms_level_dbfs", "crest_factor_db"],
            "properties": {
                "rms_level_dbfs": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "unit", "meaning"],
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"const": "dBFS"},
                        "meaning": {"const": "RMS loudness proxy; not LUFS"},
                    },
                },
                "crest_factor_db": _value_unit_schema("dB", minimum=0.0),
            },
        },
        "spectrum": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "spectral_centroid_hz",
                "spectral_bandwidth_hz",
                "spectral_rolloff_85_hz",
                "brightness_high_band_energy_share",
                "six_band_energy_shares",
            ],
            "properties": {
                "spectral_centroid_hz": _value_unit_schema("Hz", minimum=0.0),
                "spectral_bandwidth_hz": _value_unit_schema("Hz", minimum=0.0),
                "spectral_rolloff_85_hz": _value_unit_schema("Hz", minimum=0.0),
                "brightness_high_band_energy_share": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "unit", "cutoff_hz"],
                    "properties": {
                        "value": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "unit": {"const": "percent"},
                        "cutoff_hz": {"const": 3000.0},
                    },
                },
                "six_band_energy_shares": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "prefixItems": _BAND_SUMMARY_SCHEMAS,
                    "items": False,
                },
            },
        },
        "stereo": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["available", "width_side_fraction", "left_right_correlation"],
                    "properties": {
                        "available": {"const": True},
                        "width_side_fraction": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value", "unit", "range"],
                            "properties": {
                                "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "unit": {"const": "ratio"},
                                "range": {"const": [0.0, 1.0]},
                            },
                        },
                        "left_right_correlation": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value", "unit", "range"],
                            "properties": {
                                "value": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "unit": {"const": "correlation"},
                                "range": {"const": [-1.0, 1.0]},
                            },
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["available", "width_side_fraction", "left_right_correlation", "note"],
                    "properties": {
                        "available": {"const": False},
                        "width_side_fraction": {"const": None},
                        "left_right_correlation": {"const": None},
                        "note": {
                            "enum": [
                                "not reported for mono input",
                                "not reported for silent stereo input",
                            ]
                        },
                    },
                },
            ]
        },
        "analysis_counts": {
            "type": "object",
            "additionalProperties": False,
            "required": ["samples", "channels", "spectral_frames"],
            "properties": {
                "samples": {"type": "integer", "minimum": 1},
                "channels": {"type": "integer", "minimum": 1, "maximum": 2},
                "spectral_frames": {"type": "integer", "minimum": 1},
            },
        },
    },
}
STRUCTURE_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "section_scope",
        "bpm",
        "n_beats",
        "n_downbeats",
        "downbeats_seconds",
        "available_sections",
        "section_bar_boundaries",
    ],
    "properties": {
        "section_scope": {"enum": ["selected_only", "complete_track"]},
        "bpm": {"type": "number", "exclusiveMinimum": 0.0},
        "n_beats": {"type": "integer", "minimum": 0},
        "n_downbeats": {"type": "integer", "minimum": 0},
        "downbeats_seconds": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0},
        },
        "available_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "occurrence", "start_sec", "end_sec"],
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 64},
                    "occurrence": {"type": "integer", "minimum": 1},
                    "start_sec": {"type": "number", "minimum": 0.0},
                    "end_sec": {"type": "number", "exclusiveMinimum": 0.0},
                },
            },
        },
        "section_bar_boundaries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "label",
                    "occurrence",
                    "section_start_sec",
                    "section_end_sec",
                    "bar_count",
                    "bars",
                ],
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 64},
                    "occurrence": {"type": "integer", "minimum": 1},
                    "section_start_sec": {"type": "number", "minimum": 0.0},
                    "section_end_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    "bar_count": {"type": "integer", "minimum": 1},
                    "bars": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["bar_index", "start_sec", "end_sec"],
                            "properties": {
                                "bar_index": {"type": "integer", "minimum": 1},
                                "start_sec": {"type": "number", "minimum": 0.0},
                                "end_sec": {"type": "number", "exclusiveMinimum": 0.0},
                            },
                        },
                    },
                },
            },
        },
    },
}


class MultiEndpointPlanError(ValueError):
    pass


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_id: str
    name: str
    role: str
    normalized_parameters: Any | None
    dsp_summary: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiEndpointPlanError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MultiEndpointPlanError(f"{where} must be a finite number")
    return result


def _unit(value: Any, where: str) -> float:
    result = _finite_number(value, where)
    if not 0.0 <= result <= 1.0:
        raise MultiEndpointPlanError(f"{where} must be inside [0,1]")
    return result


def _finite_json(value: Any, where: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise MultiEndpointPlanError(f"{where} must be finite JSON data") from error


def _reject_runtime_fields(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise MultiEndpointPlanError(f"{where} keys must be strings")
            lowered = key.lower()
            unsafe_fragment = next(
                (fragment for fragment in _RUNTIME_ONLY_KEY_FRAGMENTS if fragment in lowered),
                None,
            )
            if lowered in _RUNTIME_ONLY_KEYS or (
                unsafe_fragment is not None and child is not False
            ):
                raise MultiEndpointPlanError(
                    f"{where} contains runtime-only field {key!r}"
                )
            _reject_runtime_fields(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_runtime_fields(child, f"{where}[{index}]")


def _validate_dsp_summary(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiEndpointPlanError(f"{where} must be an object")
    result = deepcopy(dict(value))
    try:
        jsonschema.Draft202012Validator(DSP_SUMMARY_SCHEMA).validate(result)
    except jsonschema.ValidationError as error:
        raise MultiEndpointPlanError(
            f"{where} is not an exact summarize_audio record: {error.message}"
        ) from error
    _finite_json(result, where)
    _reject_runtime_fields(result, where)
    return result


def _validate_structure_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiEndpointPlanError("structure_summary must be an object")
    result = deepcopy(dict(value))
    try:
        jsonschema.Draft202012Validator(STRUCTURE_SUMMARY_SCHEMA).validate(result)
    except jsonschema.ValidationError as error:
        raise MultiEndpointPlanError(
            f"structure_summary does not match exact music-structure schema: {error.message}"
        ) from error
    _finite_json(result, "structure_summary")
    _reject_runtime_fields(result, "structure_summary")
    downbeats = [float(value) for value in result["downbeats_seconds"]]
    if len(downbeats) != int(result["n_downbeats"]) or any(
        right <= left for left, right in zip(downbeats, downbeats[1:])
    ):
        raise MultiEndpointPlanError(
            "structure_summary downbeats must match n_downbeats and increase strictly"
        )
    if int(result["n_beats"]) < int(result["n_downbeats"]):
        raise MultiEndpointPlanError("structure_summary n_beats cannot be below n_downbeats")
    sections = result["available_sections"]
    bar_sections = result["section_bar_boundaries"]
    if len(sections) != len(bar_sections):
        raise MultiEndpointPlanError(
            "structure_summary section and bar-boundary counts differ"
        )
    identities = [(row["label"], int(row["occurrence"])) for row in sections]
    if len(set(identities)) != len(identities):
        raise MultiEndpointPlanError("section label and occurrence identities must be unique")
    previous_section_end = None
    occurrences: dict[str, list[int]] = {}
    for section in sections:
        start = float(section["start_sec"])
        if previous_section_end is not None and start < previous_section_end:
            raise MultiEndpointPlanError("sections must be chronological and non-overlapping")
        previous_section_end = float(section["end_sec"])
        occurrences.setdefault(str(section["label"]), []).append(int(section["occurrence"]))
    if (
        result["section_scope"] == "selected_only"
        and len(sections) != 1
    ):
        raise MultiEndpointPlanError("selected_only structure must contain exactly one section")
    if (
        result["section_scope"] == "complete_track"
        and any(values != list(range(1, len(values) + 1)) for values in occurrences.values())
    ):
        raise MultiEndpointPlanError("section occurrences must be sequential per label")
    downbeat_set = set(downbeats)
    for section, bar_section in zip(sections, bar_sections, strict=True):
        for label, where in (
            (section["label"], "available section label"),
            (bar_section["label"], "bar-boundary section label"),
        ):
            if not isinstance(label, str) or not _ENDPOINT_NAME.fullmatch(label):
                raise MultiEndpointPlanError(f"{where} must be a short plain phrase")
            lowered = label.lower()
            if any(token in lowered for token in _FORBIDDEN_ENDPOINT_NAME_TOKENS):
                raise MultiEndpointPlanError(f"{where} contains forbidden control or DSP text")
        identity = (section["label"], int(section["occurrence"]))
        if identity != (bar_section["label"], int(bar_section["occurrence"])):
            raise MultiEndpointPlanError("section bar-boundary identity mismatch")
        if not float(section["end_sec"]) > float(section["start_sec"]):
            raise MultiEndpointPlanError("section end must follow section start")
        if (
            float(bar_section["section_start_sec"]) != float(section["start_sec"])
            or float(bar_section["section_end_sec"]) != float(section["end_sec"])
            or int(bar_section["bar_count"]) != len(bar_section["bars"])
        ):
            raise MultiEndpointPlanError("section bar-boundary metadata mismatch")
        previous_end = None
        for expected_index, bar in enumerate(bar_section["bars"], start=1):
            start = float(bar["start_sec"])
            stop = float(bar["end_sec"])
            if int(bar["bar_index"]) != expected_index or stop <= start:
                raise MultiEndpointPlanError("bar indices or bounds are invalid")
            if start < float(section["start_sec"]) or stop > float(section["end_sec"]):
                raise MultiEndpointPlanError("bar bounds must stay inside their section")
            if start not in downbeat_set or stop not in downbeat_set:
                raise MultiEndpointPlanError("every bar boundary must match the downbeat grid")
            if previous_end is not None and start != previous_end:
                raise MultiEndpointPlanError("bar boundaries must be contiguous")
            previous_end = stop
        if (
            float(bar_section["bars"][0]["start_sec"])
            != float(section["start_sec"])
            or previous_end != float(section["end_sec"])
        ):
            raise MultiEndpointPlanError(
                "bar boundaries must cover their section from start to end"
            )
    return result


def _parameter_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def _summary_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_endpoint_name(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise MultiEndpointPlanError(f"{where} must be a string")
    result = value.strip()
    if not result or len(result) > 64 or not _ENDPOINT_NAME.fullmatch(result):
        raise MultiEndpointPlanError(
            f"{where} must be a short plain display phrase without control punctuation"
        )
    lowered = result.lower()
    forbidden = next(
        (token for token in _FORBIDDEN_ENDPOINT_NAME_TOKENS if token in lowered),
        None,
    )
    if forbidden is not None:
        raise MultiEndpointPlanError(
            f"{where} contains forbidden control or DSP token {forbidden!r}"
        )
    return result


class EndpointRegistry:
    def __init__(self, mode: str, endpoints: Iterable[EndpointSpec]) -> None:
        self.mode = mode
        count = bridge.parameter_count(mode)
        records: dict[str, EndpointSpec] = {}
        runtime_parameters: dict[str, torch.Tensor | None] = {}
        dry_count = 0

        for raw in endpoints:
            if not isinstance(raw, EndpointSpec):
                raise MultiEndpointPlanError("endpoints must be EndpointSpec values")
            endpoint_id = raw.endpoint_id
            if not isinstance(endpoint_id, str) or not _ENDPOINT_ID.fullmatch(endpoint_id):
                raise MultiEndpointPlanError(f"invalid endpoint_id {endpoint_id!r}")
            if endpoint_id in records:
                raise MultiEndpointPlanError(f"duplicate endpoint_id {endpoint_id!r}")
            endpoint_name = _validate_endpoint_name(raw.name, f"{endpoint_id}.name")
            if raw.role not in ENDPOINT_ROLES:
                raise MultiEndpointPlanError(
                    f"{endpoint_id}.role must be one of {ENDPOINT_ROLES}"
                )
            summary = _validate_dsp_summary(
                raw.dsp_summary, f"{endpoint_id}.dsp_summary"
            )

            if raw.role == "dry":
                dry_count += 1
                if raw.normalized_parameters is not None:
                    raise MultiEndpointPlanError(
                        f"dry endpoint {endpoint_id!r} must not contain parameters"
                    )
                parameters = None
            else:
                if raw.normalized_parameters is None:
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.normalized_parameters is required"
                    )
                try:
                    parameters = torch.as_tensor(raw.normalized_parameters)
                except (TypeError, ValueError, RuntimeError) as error:
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.normalized_parameters is invalid"
                    ) from error
                if tuple(parameters.shape) != (1, count):
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.normalized_parameters must have shape (1,{count})"
                    )
                if not parameters.is_floating_point() or not bool(
                    torch.isfinite(parameters).all()
                ):
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.normalized_parameters must be finite floating-point"
                    )
                if bool(((parameters < 0.0) | (parameters > 1.0)).any()):
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.normalized_parameters left [0,1]"
                    )
                parameters = parameters.detach().clone()

            if not isinstance(raw.provenance, Mapping) or not raw.provenance:
                raise MultiEndpointPlanError(
                    f"{endpoint_id}.provenance must be a non-empty object"
                )
            provenance = deepcopy(dict(raw.provenance))
            _finite_json(provenance, f"{endpoint_id}.provenance")
            if raw.role != "dry":
                expected_parameter_hash = _parameter_sha256(parameters)
                required_attestation = {
                    "verification_status": "verified",
                    "source_artifacts_verified": True,
                    "normalized_parameters_sha256": expected_parameter_hash,
                    "dsp_summary_sha256": _summary_sha256(summary),
                }
                for name, expected in required_attestation.items():
                    if provenance.get(name) != expected:
                        raise MultiEndpointPlanError(
                            f"{endpoint_id}.provenance {name} is not a verified attestation"
                        )
                if not isinstance(provenance.get("verification_method"), str) or not provenance[
                    "verification_method"
                ].strip():
                    raise MultiEndpointPlanError(
                        f"{endpoint_id}.provenance verification_method must be non-empty"
                    )
            records[endpoint_id] = EndpointSpec(
                endpoint_id=endpoint_id,
                name=endpoint_name,
                role=raw.role,
                normalized_parameters=None,
                dsp_summary=summary,
                provenance=provenance,
            )
            runtime_parameters[endpoint_id] = parameters

        if not records:
            raise MultiEndpointPlanError("endpoint registry must not be empty")
        if dry_count != 1:
            raise MultiEndpointPlanError("endpoint registry needs exactly one dry endpoint")
        self._records = records
        self._runtime_parameters = runtime_parameters

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def dry_endpoint_id(self) -> str:
        return next(
            endpoint_id
            for endpoint_id, record in self._records.items()
            if record.role == "dry"
        )

    def role(self, endpoint_id: str) -> str:
        try:
            return self._records[endpoint_id].role
        except KeyError as error:
            raise MultiEndpointPlanError(f"unknown endpoint_id {endpoint_id!r}") from error

    def parameters(self, endpoint_id: str) -> torch.Tensor | None:
        if endpoint_id not in self._records:
            raise MultiEndpointPlanError(f"unknown endpoint_id {endpoint_id!r}")
        value = self._runtime_parameters[endpoint_id]
        return None if value is None else value.clone()

    def llm_view(self) -> list[dict[str, Any]]:
        return [
            {
                "endpoint_id": endpoint_id,
                "name": record.name,
                "role": record.role,
                "dsp_summary": deepcopy(dict(record.dsp_summary)),
            }
            for endpoint_id, record in self._records.items()
        ]

    def provenance_status(self) -> dict[str, Any]:
        return {
            "records_present": {
                endpoint_id: bool(self._records[endpoint_id].provenance)
                for endpoint_id in self.endpoint_ids
                if self.role(endpoint_id) != "dry"
            },
            "normalized_parameter_hashes_verified_here": True,
            "source_artifact_verification_attested": True,
            "source_artifacts_reverified_here": False,
            "required_outer_runner_action": (
                "verify source hashes and reproduce endpoint parameters before execution"
            ),
        }


def _event_schema(endpoint_ids: tuple[str, ...]) -> dict[str, Any]:
    common = {
        "id": {"type": "string", "minLength": 1, "maxLength": 32},
        "instruction_evidence": {"type": "string", "minLength": 1},
        "duration_weight": {"type": "number", "exclusiveMinimum": 0.0},
        "handle1_time_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "handle2_time_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "handle1_value_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "handle2_value_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(EVENT_FIELDS),
                "properties": {
                    **common,
                    "relation": {"const": "set_state"},
                    "source_endpoint_id": {"enum": list(endpoint_ids)},
                    "target_endpoint_id": {"enum": list(endpoint_ids)},
                    "target_amount": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(EVENT_FIELDS),
                "properties": {
                    **common,
                    "relation": {"const": "hold_current"},
                    "source_endpoint_id": {"const": None},
                    "target_endpoint_id": {"const": None},
                    "target_amount": {"const": None},
                },
            },
        ]
    }


def make_plan_schema(
    *,
    registry: EndpointRegistry,
    allowed_stems: list[str],
    allowed_sections: list[str],
    allowed_section_occurrences: list[tuple[str, int]] | None = None,
    locked_stem: str | None = None,
    locked_section: str | None = None,
    locked_occurrence: int | None = None,
) -> dict[str, Any]:
    if not allowed_stems or not allowed_sections:
        raise MultiEndpointPlanError("allowed stems and sections must be non-empty")
    if len(set(allowed_stems)) != len(allowed_stems) or len(set(allowed_sections)) != len(allowed_sections):
        raise MultiEndpointPlanError("allowed stems and sections must be unique")
    for value, where in (
        *((value, "allowed stem") for value in allowed_stems),
        *((value, "allowed section") for value in allowed_sections),
    ):
        _validate_endpoint_name(value, where)
    pairs = (
        [(section, 1) for section in allowed_sections]
        if allowed_section_occurrences is None
        else list(allowed_section_occurrences)
    )
    if not pairs or len(set(pairs)) != len(pairs):
        raise MultiEndpointPlanError("allowed section-occurrence pairs must be non-empty and unique")
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or pair[0] not in allowed_sections
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], (int, np.integer))
            or int(pair[1]) < 1
        ):
            raise MultiEndpointPlanError("allowed section-occurrence pair is invalid")
    if locked_stem is not None and locked_stem not in allowed_stems:
        raise MultiEndpointPlanError("locked stem is unavailable")
    if locked_section is not None and locked_section not in allowed_sections:
        raise MultiEndpointPlanError("locked section is unavailable")
    if locked_occurrence is not None and (
        isinstance(locked_occurrence, bool)
        or not isinstance(locked_occurrence, (int, np.integer))
        or int(locked_occurrence) < 1
    ):
        raise MultiEndpointPlanError("locked occurrence must be a positive integer")
    if (
        locked_section is not None
        and locked_occurrence is not None
        and (locked_section, int(locked_occurrence)) not in pairs
    ):
        raise MultiEndpointPlanError("locked section-occurrence pair is unavailable")
    endpoint_ids = registry.endpoint_ids
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FlexibleMultiEndpointAutomationPlanV1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "target_stem",
            "effect_mode",
            "section",
            "occurrence",
            "initial_endpoint_id",
            "events",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "target_stem": (
                {"const": locked_stem}
                if locked_stem is not None
                else {"enum": allowed_stems}
            ),
            "effect_mode": {"const": registry.mode},
            "section": (
                {"const": locked_section}
                if locked_section is not None
                else {"enum": allowed_sections}
            ),
            "occurrence": (
                {"const": int(locked_occurrence)}
                if locked_occurrence is not None
                else {"type": "integer", "minimum": 1}
            ),
            "initial_endpoint_id": {"enum": list(endpoint_ids)},
            "events": {
                "type": "array",
                "minItems": 0,
                "maxItems": MAX_EVENTS,
                "items": _event_schema(endpoint_ids),
            },
        },
    }
    if locked_section is None or locked_occurrence is None:
        schema["allOf"] = [
            {
                "oneOf": [
                    {
                        "properties": {
                            "section": {"const": section},
                            "occurrence": {"const": int(occurrence)},
                        },
                        "required": ["section", "occurrence"],
                    }
                    for section, occurrence in pairs
                ]
            }
        ]
    return schema


def validate_plan(
    raw: Any,
    *,
    schema: Mapping[str, Any],
    registry: EndpointRegistry,
    instruction: str | None = None,
) -> dict[str, Any]:
    try:
        jsonschema.Draft202012Validator(dict(schema)).validate(raw)
    except jsonschema.ValidationError as error:
        raise MultiEndpointPlanError(f"plan schema validation failed: {error.message}") from error
    if not isinstance(raw, Mapping):
        raise MultiEndpointPlanError("plan must be an object")

    endpoint_index = {
        endpoint_id: index for index, endpoint_id in enumerate(registry.endpoint_ids)
    }
    initial = str(raw["initial_endpoint_id"])
    current_state = np.zeros(len(registry.endpoint_ids), dtype=np.float64)
    current_state[endpoint_index[initial]] = 1.0
    seen: set[str] = set()
    evidence_cursor = 0
    clean_events: list[dict[str, Any]] = []
    for index, event in enumerate(raw["events"]):
        where = f"events[{index}]"
        if set(event) != set(EVENT_FIELDS):
            raise MultiEndpointPlanError(f"{where} fields do not match contract")
        event_id = event["id"]
        if event_id in seen:
            raise MultiEndpointPlanError(f"duplicate event id {event_id!r}")
        evidence = event["instruction_evidence"]
        if instruction is not None:
            position = instruction.find(evidence, evidence_cursor)
            if position < 0:
                raise MultiEndpointPlanError(
                    f"{where}.instruction_evidence is not ordered evidence"
                )
            evidence_cursor = position + len(evidence)

        relation = event["relation"]
        source = event["source_endpoint_id"]
        target = event["target_endpoint_id"]
        raw_amount = event["target_amount"]
        if relation == "set_state":
            if source == target:
                raise MultiEndpointPlanError(
                    f"{where} source and target endpoints must differ"
                )
            if source not in endpoint_index or target not in endpoint_index:
                raise MultiEndpointPlanError(f"{where} uses unknown endpoint")
            target_amount = _unit(raw_amount, f"{where}.target_amount")
            desired_state = np.zeros_like(current_state)
            desired_state[endpoint_index[str(source)]] = 1.0 - target_amount
            desired_state[endpoint_index[str(target)]] += target_amount
            if np.array_equal(desired_state, current_state):
                raise MultiEndpointPlanError(
                    f"{where} desired state already equals current state; use hold_current"
                )
        elif relation == "hold_current":
            if source is not None or target is not None or raw_amount is not None:
                raise MultiEndpointPlanError(
                    f"{where} endpoint IDs and target_amount must be null for hold_current"
                )
            target_amount = None
            desired_state = current_state.copy()
        else:
            raise MultiEndpointPlanError(f"{where}.relation is invalid")

        duration = _finite_number(event["duration_weight"], f"{where}.duration_weight")
        if duration <= 0.0:
            raise MultiEndpointPlanError(f"{where}.duration_weight must be positive")
        clean_events.append(
            {
                "id": event_id,
                "instruction_evidence": evidence,
                "relation": relation,
                "source_endpoint_id": source,
                "target_endpoint_id": target,
                "target_amount": target_amount,
                "duration_weight": duration,
                "handle1_time_fraction": _unit(
                    event["handle1_time_fraction"], f"{where}.handle1_time_fraction"
                ),
                "handle2_time_fraction": _unit(
                    event["handle2_time_fraction"], f"{where}.handle2_time_fraction"
                ),
                "handle1_value_fraction": _unit(
                    event["handle1_value_fraction"], f"{where}.handle1_value_fraction"
                ),
                "handle2_value_fraction": _unit(
                    event["handle2_value_fraction"], f"{where}.handle2_value_fraction"
                ),
            }
        )
        current_state = desired_state
        seen.add(event_id)
    return {**dict(raw), "events": clean_events}


def _bezier(c0: float, c1: float, c2: float, c3: float, u: np.ndarray) -> np.ndarray:
    one_minus = 1.0 - u
    return (
        one_minus**3 * c0
        + 3.0 * one_minus**2 * u * c1
        + 3.0 * one_minus * u**2 * c2
        + u**3 * c3
    )


def _event_progress(event: Mapping[str, Any], local_time: np.ndarray) -> np.ndarray:
    x1 = float(event["handle1_time_fraction"])
    x2 = x1 + (1.0 - x1) * float(event["handle2_time_fraction"])
    y1 = float(event["handle1_value_fraction"])
    y2 = y1 + (1.0 - y1) * float(event["handle2_value_fraction"])
    low = np.zeros_like(local_time)
    high = np.ones_like(local_time)
    for _ in range(60):
        middle = 0.5 * (low + high)
        x_value = _bezier(0.0, x1, x2, 1.0, middle)
        low = np.where(x_value < local_time, middle, low)
        high = np.where(x_value < local_time, high, middle)
    result = _bezier(0.0, y1, y2, 1.0, 0.5 * (low + high))
    result[local_time <= 0.0] = 0.0
    result[local_time >= 1.0] = 1.0
    return result


def evaluate_endpoint_schedule(
    plan: Mapping[str, Any],
    *,
    registry: EndpointRegistry,
    n_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(n_samples, int) or n_samples < 1:
        raise MultiEndpointPlanError("n_samples must be a positive integer")
    endpoint_ids = registry.endpoint_ids
    index_by_id = {endpoint_id: index for index, endpoint_id in enumerate(endpoint_ids)}
    initial = str(plan["initial_endpoint_id"])
    if initial not in index_by_id:
        raise MultiEndpointPlanError("initial endpoint is not registered")
    events = list(plan["events"])
    weights = np.zeros((len(endpoint_ids), n_samples), dtype=np.float64)
    minimum_intervals = np.asarray(
        [2 if event["relation"] == "set_state" else 1 for event in events],
        dtype=np.int64,
    )
    required_intervals = int(minimum_intervals.sum())
    if events and n_samples - 1 < required_intervals:
        raise MultiEndpointPlanError(
            "active interval is too short: every transition needs at least two "
            "sample intervals and every hold needs one"
        )
    if not events:
        weights[index_by_id[initial], :] = 1.0
        return weights, {
            "endpoint_ids": list(endpoint_ids),
            "initial_endpoint_id": initial,
            "final_endpoint_id": initial,
            "final_state_weights": {initial: 1.0},
            "events": [],
            "continuity_by_construction": True,
            "weights_sum_to_one": True,
        }

    durations = np.asarray([event["duration_weight"] for event in events], dtype=np.float64)
    maximum_duration = float(durations.max())
    scaled_durations = durations / maximum_duration
    scaled_total = float(scaled_durations.sum())
    if not math.isfinite(scaled_total) or scaled_total <= 0.0:
        raise MultiEndpointPlanError("duration weights cannot be normalized safely")
    requested_fractions = scaled_durations / scaled_total

    total_intervals = n_samples - 1
    remaining_intervals = total_intervals - required_intervals
    desired_extra = requested_fractions * remaining_intervals
    extra_intervals = np.floor(desired_extra).astype(np.int64)
    unassigned = int(remaining_intervals - int(extra_intervals.sum()))
    if unassigned:
        remainders = desired_extra - extra_intervals
        order = np.argsort(-remainders, kind="stable")
        extra_intervals[order[:unassigned]] += 1
    event_intervals = extra_intervals + minimum_intervals
    if int(event_intervals.sum()) != total_intervals or bool(
        (event_intervals < minimum_intervals).any()
    ):
        raise MultiEndpointPlanError("event interval allocation failed")

    current_state = np.zeros(len(endpoint_ids), dtype=np.float64)
    current_state[index_by_id[initial]] = 1.0

    def plain_state(value: np.ndarray) -> dict[str, float]:
        return {
            endpoint_id: float(value[index])
            for endpoint_id, index in index_by_id.items()
            if float(value[index]) > 0.0
        }

    report_events: list[dict[str, Any]] = []
    cursor = 0
    for event_index, (event, interval_count) in enumerate(
        zip(events, event_intervals, strict=True)
    ):
        interval_count = int(interval_count)
        stop_sample = cursor + interval_count
        sample_slice = slice(cursor, stop_sample + 1)
        local_time = np.arange(interval_count + 1, dtype=np.float64) / interval_count
        if event["relation"] == "hold_current":
            desired_state = current_state.copy()
            weights[:, sample_slice] = current_state[:, None]
        else:
            source = str(event["source_endpoint_id"])
            target = str(event["target_endpoint_id"])
            amount = float(event["target_amount"])
            desired_state = np.zeros_like(current_state)
            desired_state[index_by_id[source]] = 1.0 - amount
            desired_state[index_by_id[target]] += amount
            progress = _event_progress(event, local_time)
            weights[:, sample_slice] = (
                current_state[:, None]
                + (desired_state - current_state)[:, None] * progress[None, :]
            )
            weights[:, stop_sample] = desired_state
        report_events.append(
            {
                "id": event["id"],
                "relation": event["relation"],
                "blend_source_endpoint_id": event["source_endpoint_id"],
                "blend_target_endpoint_id": event["target_endpoint_id"],
                "target_amount": event["target_amount"],
                "start_state_weights": plain_state(current_state),
                "desired_state_weights": plain_state(desired_state),
                "start_sample": cursor,
                "stop_sample_inclusive": stop_sample,
                "allocated_intervals": interval_count,
                "allocated_samples_including_shared_boundaries": interval_count + 1,
                "normalized_start_time": cursor / total_intervals,
                "normalized_stop_time": stop_sample / total_intervals,
                "requested_duration_fraction": float(requested_fractions[event_index]),
                "rendered_duration_fraction": interval_count / total_intervals,
                "desired_state_has_exact_sample_support": bool(
                    np.array_equal(weights[:, stop_sample], desired_state)
                ),
            }
        )
        current_state = desired_state
        cursor = stop_sample

    if cursor != n_samples - 1:
        raise MultiEndpointPlanError("event schedule did not cover the active interval")

    if not bool(np.isfinite(weights).all()) or float(weights.min()) < -1e-12:
        raise MultiEndpointPlanError("resolved endpoint weights are invalid")
    sums = weights.sum(axis=0)
    if not bool(np.allclose(sums, 1.0, rtol=0.0, atol=1e-10)):
        raise MultiEndpointPlanError("resolved endpoint weights do not sum to one")
    nonzero_final = np.flatnonzero(current_state == 1.0)
    final_endpoint_id = (
        endpoint_ids[int(nonzero_final[0])]
        if nonzero_final.size == 1 and np.count_nonzero(current_state) == 1
        else None
    )
    return weights, {
        "endpoint_ids": list(endpoint_ids),
        "initial_endpoint_id": initial,
        "final_endpoint_id": final_endpoint_id,
        "final_state_weights": plain_state(current_state),
        "events": report_events,
        "requested_duration_fractions": requested_fractions.tolist(),
        "allocated_intervals": event_intervals.tolist(),
        "duration_normalization": "divide by maximum before summing; no overflow",
        "minimum_event_support": (
            "two intervals for transition, one for hold, exact destination sample"
        ),
        "continuity_by_construction": True,
        "weights_sum_to_one": True,
        "sorted": False,
        "clamped": False,
        "repaired": False,
    }
