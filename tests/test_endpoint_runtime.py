from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import dsp_summary
import endpoint_runtime as runtime
import planner
import reference_bridge as bridge

SAMPLE_RATE = 44_100


def _sha_array(value: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def _sha_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _summary(audio: torch.Tensor) -> dict:
    value = np.ascontiguousarray(audio[0].T.detach().cpu().numpy())
    return dsp_summary.summarize_audio(value, SAMPLE_RATE)


def _spec(
    endpoint_id: str,
    *,
    role: str,
    endpoint_mode: str | None,
    parameters: torch.Tensor | None,
    renderer_mode: str,
    summary: dict,
) -> runtime.MaskedEndpointSpec:
    attestation = runtime.applicability_attestation(
        endpoint_id=endpoint_id,
        endpoint_mode=endpoint_mode,
        renderer_mode=renderer_mode,
    )
    if parameters is None:
        provenance = {"verification_method": "test identity", **attestation}
    else:
        embedded = runtime.embed_endpoint_parameters(
            parameters, endpoint_mode=endpoint_mode, renderer_mode=renderer_mode
        )
        provenance = {
            "verification_status": "verified",
            "verification_method": "test source replay",
            "source_artifacts_verified": True,
            "normalized_parameters_sha256": _sha_array(embedded),
            "dsp_summary_sha256": _sha_json(summary),
            **attestation,
        }
    return runtime.MaskedEndpointSpec(
        endpoint_id=endpoint_id,
        role=role,
        endpoint_mode=endpoint_mode,
        normalized_parameters=parameters,
        dsp_summary=summary,
        provenance=provenance,
    )


def _registry(
    renderer_mode: str,
    audio: torch.Tensor,
    endpoints: list[tuple[str, str, str, torch.Tensor]],
) -> runtime.MaskedEndpointRegistry:
    summary = _summary(audio)
    specs = [
        _spec(
            "endpoint_000",
            role="dry",
            endpoint_mode=None,
            parameters=None,
            renderer_mode=renderer_mode,
            summary=summary,
        )
    ]
    for endpoint_id, role, endpoint_mode, parameters in endpoints:
        specs.append(
            _spec(
                endpoint_id,
                role=role,
                endpoint_mode=endpoint_mode,
                parameters=parameters,
                renderer_mode=renderer_mode,
                summary=summary,
            )
        )
    return runtime.MaskedEndpointRegistry(renderer_mode, specs)


def _plan_and_schema(
    registry: runtime.MaskedEndpointRegistry,
    instruction: str,
    *,
    initial: str = "endpoint_000",
    target: str = "endpoint_001",
) -> tuple[dict, dict]:
    schema = planner.make_plan_schema(
        registry=registry,
        allowed_stems=["vocals"],
        allowed_sections=["unit"],
        allowed_section_occurrences=[("unit", 1)],
        locked_stem="vocals",
        locked_section="unit",
        locked_occurrence=1,
    )
    plan = {
        "schema_version": planner.SCHEMA_VERSION,
        "target_stem": "vocals",
        "effect_mode": registry.mode,
        "section": "unit",
        "occurrence": 1,
        "initial_endpoint_id": initial,
        "events": [
            {
                "id": "event_1",
                "instruction_evidence": instruction,
                "relation": "set_state",
                "source_endpoint_id": initial,
                "target_endpoint_id": target,
                "target_amount": 1.0,
                "duration_weight": 1.0,
                "handle1_time_fraction": 0.25,
                "handle2_time_fraction": 0.75,
                "handle1_value_fraction": 0.25,
                "handle2_value_fraction": 0.75,
            }
        ],
    }
    return plan, schema


def _render(
    audio: torch.Tensor,
    registry: runtime.MaskedEndpointRegistry,
    instruction: str,
) -> tuple[torch.Tensor, dict]:
    plan, schema = _plan_and_schema(registry, instruction)
    return runtime.MaskedMultiEndpointRenderer(sample_rate=SAMPLE_RATE).render_plan(
        audio,
        registry=registry,
        validated_plan=plan,
        plan_schema=schema,
        instruction=instruction,
        active_start_sample=1_000,
        active_stop_sample=2_000,
        attack_start_sample=500,
        release_stop_sample=2_500,
    )


def test_eq_native_and_masked_joint_are_bit_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    audio = 0.01 * torch.randn(1, 2, 70_000, generator=generator)
    parameters = torch.full((1, bridge.parameter_count("eq")), 0.37)
    native = _registry(
        "eq", audio, [("endpoint_001", "caller_natural_language", "eq", parameters)]
    )
    joint = _registry(
        "eq_reverb",
        audio,
        [("endpoint_001", "caller_natural_language", "eq", parameters)],
    )
    instruction = "nonce_eq_transition"
    native_output, _ = _render(audio, native, instruction)
    joint_output, report = _render(audio, joint, instruction)
    assert torch.equal(native_output, joint_output)
    assert report["reverb"]["effect_applicability"]["applicable_endpoint_ids"] == []
    assert report["reverb"]["effect_applicability"]["excluded_endpoint_ids"] == [
        "endpoint_001"
    ]


def test_reverb_native_and_masked_joint_are_bit_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    audio = 0.005 * torch.randn(1, 2, 70_000, generator=generator)
    parameters = torch.full((1, bridge.parameter_count("reverb")), 0.23)
    native = _registry(
        "reverb",
        audio,
        [("endpoint_001", "processed_reference", "reverb", parameters)],
    )
    joint = _registry(
        "eq_reverb",
        audio,
        [("endpoint_001", "processed_reference", "reverb", parameters)],
    )
    instruction = "nonce_reverb_transition"
    native_output, _ = _render(audio, native, instruction)
    joint_output, report = _render(audio, joint, instruction)
    assert torch.equal(native_output, joint_output)
    assert report["eq"]["effect_applicability"]["applicable_endpoint_ids"] == []
    assert report["eq"]["effect_applicability"]["excluded_endpoint_ids"] == [
        "endpoint_001"
    ]


def test_mixed_registry_filters_each_effect_without_parameter_leak() -> None:
    audio = torch.zeros(1, 2, 70_000)
    audio[:, :, :3_000] = 0.01
    eq_parameters = torch.full((1, bridge.parameter_count("eq")), 0.41)
    reverb_parameters = torch.full((1, bridge.parameter_count("reverb")), 0.19)
    registry = _registry(
        "eq_reverb",
        audio,
        [
            ("endpoint_001", "caller_natural_language", "eq", eq_parameters),
            ("endpoint_002", "processed_reference", "reverb", reverb_parameters),
        ],
    )
    weights = torch.zeros(3, audio.shape[-1])
    weights[1] = 0.4
    weights[2] = 0.6
    activation = torch.ones(audio.shape[-1])
    renderer = runtime.MaskedMultiEndpointRenderer(sample_rate=SAMPLE_RATE)
    after_eq, eq_report = renderer._render_multi_eq(
        audio, registry, weights, activation
    )
    _, reverb_report = renderer._render_multi_reverb(
        after_eq,
        registry,
        weights,
        activation,
        parameter_start=bridge.parameter_count("eq"),
    )
    assert eq_report["effect_applicability"]["applicable_endpoint_ids"] == [
        "endpoint_001"
    ]
    assert eq_report["effect_applicability"]["excluded_endpoint_ids"] == [
        "endpoint_002"
    ]
    assert reverb_report["effect_applicability"]["applicable_endpoint_ids"] == [
        "endpoint_002"
    ]
    assert reverb_report["effect_applicability"]["excluded_endpoint_ids"] == [
        "endpoint_001"
    ]
    assert eq_report["effect_applicability"]["excluded_parameters_used"] is False
    assert reverb_report["effect_applicability"]["excluded_parameters_used"] is False


def test_tampered_effect_mask_is_rejected() -> None:
    audio = torch.zeros(1, 2, 2_048)
    summary = _summary(audio)
    parameters = torch.full((1, bridge.parameter_count("eq")), 0.5)
    spec = _spec(
        "endpoint_001",
        role="caller_natural_language",
        endpoint_mode="eq",
        parameters=parameters,
        renderer_mode="eq_reverb",
        summary=summary,
    )
    changed = dict(spec.provenance)
    changed["effect_applicability"] = {"eq": True, "reverb": True}
    bad = runtime.MaskedEndpointSpec(
        endpoint_id=spec.endpoint_id,
        role=spec.role,
        endpoint_mode=spec.endpoint_mode,
        normalized_parameters=spec.normalized_parameters,
        dsp_summary=spec.dsp_summary,
        provenance=changed,
    )
    dry = _spec(
        "endpoint_000",
        role="dry",
        endpoint_mode=None,
        parameters=None,
        renderer_mode="eq_reverb",
        summary=summary,
    )
    with pytest.raises(
        runtime.MaskedMultiEndpointRuntimeError,
        match="applicability differs",
    ):
        runtime.MaskedEndpointRegistry("eq_reverb", [dry, bad])


def test_effect_applicability_storage_is_immutable_and_reverified() -> None:
    audio = torch.zeros(1, 2, 2_048)
    parameters = torch.full((1, bridge.parameter_count("reverb")), 0.5)
    registry = _registry(
        "eq_reverb",
        audio,
        [("endpoint_001", "processed_reference", "reverb", parameters)],
    )
    assert registry.applicable("endpoint_001", "eq") is False
    with pytest.raises(TypeError):
        registry._endpoint_modes["endpoint_001"] = "eq"
    registry.verify_effect_applicability()
    assert registry.provenance_status()["effect_applicability_verified_here"] is True


def test_dry_only_joint_registry_is_bit_exact_identity() -> None:
    generator = torch.Generator().manual_seed(29)
    audio = 0.01 * torch.randn(1, 2, 70_000, generator=generator)
    registry = _registry("eq_reverb", audio, [])
    instruction = "nonce_identity_instruction"
    schema = planner.make_plan_schema(
        registry=registry,
        allowed_stems=["vocals"],
        allowed_sections=["unit"],
        allowed_section_occurrences=[("unit", 1)],
        locked_stem="vocals",
        locked_section="unit",
        locked_occurrence=1,
    )
    plan = {
        "schema_version": planner.SCHEMA_VERSION,
        "target_stem": "vocals",
        "effect_mode": "eq_reverb",
        "section": "unit",
        "occurrence": 1,
        "initial_endpoint_id": "endpoint_000",
        "events": [],
    }
    output, report = runtime.MaskedMultiEndpointRenderer(
        sample_rate=SAMPLE_RATE
    ).render_plan(
        audio,
        registry=registry,
        validated_plan=plan,
        plan_schema=schema,
        instruction=instruction,
        active_start_sample=1_000,
        active_stop_sample=2_000,
        attack_start_sample=500,
        release_stop_sample=2_500,
    )
    assert torch.equal(output, audio)
    assert report["eq"]["backend"] == "dry identity"
    assert report["reverb"]["active_endpoint_ids"] == []
    assert report["reverb"]["tail_expected"] is False
