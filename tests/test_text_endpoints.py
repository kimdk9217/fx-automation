from __future__ import annotations

import json

import numpy as np
import pytest

import endpoint_manifest as manifest
import reference_matcher
import text_endpoints as contract


def _endpoint(text: str, mode: str = "eq") -> manifest.CanonicalEndpoint:
    return manifest.CanonicalEndpoint(
        endpoint_id="endpoint_001",
        kind="natural_language",
        effect_mode=mode,
        user_description=text,
        source_audio=None,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


@pytest.mark.parametrize("mode,count", [("eq", 18), ("reverb", 25), ("eq_reverb", 43)])
def test_static_schema_has_no_time_or_state_fields(mode: str, count: int) -> None:
    endpoint = _endpoint("nonce static target", mode)
    request = contract.build_static_request(
        endpoint,
        target_stem="bass",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        base_url="https://api.openai.com/v1",
    )
    forbidden = {
        "trajectory", "events", "duration_weight", "target_amount",
        "initial_endpoint_id", "final_endpoint_id", "state_selector",
    }
    assert forbidden.isdisjoint(set(_walk_keys(request["schema"])))
    assert request["context"]["USER_SOUND_DESCRIPTION_EXACT"] == endpoint.user_description
    plan = {
        "schema_version": contract.STATIC_PLAN_VERSION,
        "target_stem": "bass",
        "effect_mode": mode,
        "parameters": {},
    }
    if mode in {"eq", "eq_reverb"}:
        plan["parameters"]["eq"] = {
            band: {"gain_db": 0.0, "frequency_hz": 0.5 * sum(bounds), "q_factor": 2.75}
            for band in contract.EQ_BANDS
            for bounds in [
                reference_matcher.official_eq_parameter_ranges()[f"{band}_cutoff_freq"]
            ]
        }
    if mode in {"reverb", "eq_reverb"}:
        plan["parameters"]["reverb"] = {
            "band_gains": [0.37] * 12,
            "band_decays": [0.61] * 12,
            "mix": 0.29,
        }
    normalized = contract.normalized_parameters_from_plan(
        plan, endpoint=endpoint, target_stem="bass"
    )
    assert normalized.shape == (1, count)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert ((0.0 <= normalized) & (normalized <= 1.0)).all()


def test_text_change_changes_exact_request_hash_only() -> None:
    kwargs = {
        "target_stem": "other",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "base_url": "https://api.openai.com/v1",
    }
    first = contract.build_static_request(_endpoint("nonce-one"), **kwargs)
    second = contract.build_static_request(_endpoint("nonce-two"), **kwargs)
    assert first["schema_sha256"] == second["schema_sha256"]
    assert first["native_request_sha256"] != second["native_request_sha256"]
    body = second["native_request"]["body"]
    serialized = json.dumps(body, ensure_ascii=False)
    assert "nonce-two" in serialized and "nonce-one" not in serialized


def test_wrong_stem_or_extra_timeline_field_rejected() -> None:
    endpoint = _endpoint("nonce")
    plan = {
        "schema_version": contract.STATIC_PLAN_VERSION,
        "target_stem": "vocals",
        "effect_mode": "eq",
        "parameters": {"eq": {}},
        "events": [],
    }
    with pytest.raises(contract.DynamicEndpointContractError):
        contract.validate_static_plan(plan, endpoint=endpoint, target_stem="drums")


@pytest.mark.parametrize("mode", ["eq", "reverb"])
def test_single_effect_endpoint_has_explicit_joint_bypass_embedding(mode: str) -> None:
    endpoint = _endpoint("nonce", mode)
    plan = {
        "schema_version": contract.STATIC_PLAN_VERSION,
        "target_stem": "vocals",
        "effect_mode": mode,
        "parameters": {},
    }
    if mode == "eq":
        ranges = reference_matcher.official_eq_parameter_ranges()
        plan["parameters"]["eq"] = {
            band: {
                "gain_db": 0.0,
                "frequency_hz": sum(ranges[f"{band}_cutoff_freq"]) / 2.0,
                "q_factor": sum(ranges[f"{band}_q_factor"]) / 2.0,
            }
            for band in contract.EQ_BANDS
        }
    else:
        plan["parameters"]["reverb"] = {
            "band_gains": [0.2] * 12,
            "band_decays": [0.7] * 12,
            "mix": 0.3,
        }
    embedded = contract.normalized_parameters_from_plan(
        plan,
        endpoint=endpoint,
        target_stem="vocals",
        renderer_mode="eq_reverb",
    ) if mode == "eq" else None
    report = contract.renderer_embedding_contract(
        endpoint_mode=mode, renderer_mode="eq_reverb"
    )
    assert report["style_values_injected"] is False
    if mode == "eq":
        assert embedded is not None and embedded.shape == (1, 43)
        assert report["structural_bypass_only"] is True
        assert report["execution_compatible"] is True
        assert embedded[0, -1] == 0.0
    else:
        assert report["structural_bypass_only"] is False
        assert report["execution_compatible"] is False
        assert report["requires_effect_applicability_mask"] is True
        with pytest.raises(contract.DynamicEndpointContractError, match="applicability mask"):
            contract.normalized_parameters_from_plan(
                plan,
                endpoint=endpoint,
                target_stem="vocals",
                renderer_mode="eq_reverb",
            )
