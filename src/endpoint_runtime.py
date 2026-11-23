"""Per-effect masks so EQ-only or reverb-only endpoints leave the other effect untouched."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import torch

import planner
import reference_bridge as bridge
import reference_matcher as matcher
import renderer

RUNTIME_VERSION = "masked-multi-endpoint-runtime-v3"
EFFECTS = ("eq", "reverb")


class MaskedMultiEndpointRuntimeError(ValueError):
    pass


def _sha_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effect_applicability_for_mode(mode: str) -> dict[str, bool]:
    if mode == "eq":
        return {"eq": True, "reverb": False}
    if mode == "reverb":
        return {"eq": False, "reverb": True}
    if mode == "eq_reverb":
        return {"eq": True, "reverb": True}
    raise MaskedMultiEndpointRuntimeError(f"unsupported endpoint mode {mode!r}")


def embed_endpoint_parameters(
    parameters: np.ndarray | torch.Tensor,
    *,
    endpoint_mode: str,
    renderer_mode: str,
) -> torch.Tensor:
    try:
        value = torch.as_tensor(parameters).detach().clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise MaskedMultiEndpointRuntimeError("endpoint parameters are invalid") from error
    expected = bridge.parameter_count(endpoint_mode)
    if (
        tuple(value.shape) != (1, expected)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool(((value < 0.0) | (value > 1.0)).any())
    ):
        raise MaskedMultiEndpointRuntimeError(
            f"{endpoint_mode} parameters must be finite [1,{expected}] values in [0,1]"
        )
    if renderer_mode == endpoint_mode:
        return value
    if renderer_mode != "eq_reverb" or endpoint_mode == "eq_reverb":
        raise MaskedMultiEndpointRuntimeError(
            f"cannot embed {endpoint_mode!r} into renderer {renderer_mode!r}"
        )
    if endpoint_mode == "eq":
        padding = torch.zeros(
            (1, matcher.REVERB_PARAMETER_COUNT),
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, padding), dim=1)
    if endpoint_mode == "reverb":
        padding = torch.zeros(
            (1, matcher.EQ_PARAMETER_COUNT),
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((padding, value), dim=1)
    raise MaskedMultiEndpointRuntimeError(f"unsupported endpoint mode {endpoint_mode!r}")


@dataclass(frozen=True)
class MaskedEndpointSpec:
    endpoint_id: str
    role: str
    endpoint_mode: str | None
    normalized_parameters: Any | None
    dsp_summary: Mapping[str, Any]
    provenance: Mapping[str, Any]


class MaskedEndpointRegistry:
    def __init__(
        self, renderer_mode: str, endpoints: Iterable[MaskedEndpointSpec]
    ) -> None:
        bridge.parameter_count(renderer_mode)
        specs = tuple(endpoints)
        if not specs:
            raise MaskedMultiEndpointRuntimeError("endpoint registry is empty")
        endpoint_modes: dict[str, str | None] = {}
        mask_hashes: dict[str, str] = {}
        core_specs: list[planner.EndpointSpec] = []
        for spec in specs:
            if not isinstance(spec, MaskedEndpointSpec):
                raise MaskedMultiEndpointRuntimeError(
                    "registry entries must be MaskedEndpointSpecV3"
                )
            if spec.role == "dry":
                if spec.endpoint_mode is not None or spec.normalized_parameters is not None:
                    raise MaskedMultiEndpointRuntimeError(
                        "dry endpoint has neither a processor mode nor parameters"
                    )
                mask = {"eq": False, "reverb": False}
                embedded = None
            else:
                if spec.endpoint_mode is None:
                    raise MaskedMultiEndpointRuntimeError(
                        f"{spec.endpoint_id} lacks explicit endpoint mode"
                    )
                mask = effect_applicability_for_mode(spec.endpoint_mode)
                embedded = embed_endpoint_parameters(
                    spec.normalized_parameters,
                    endpoint_mode=spec.endpoint_mode,
                    renderer_mode=renderer_mode,
                )
            mask_record = {
                "endpoint_id": spec.endpoint_id,
                "renderer_mode": renderer_mode,
                "endpoint_mode": spec.endpoint_mode,
                "applicability": mask,
            }
            provenance = dict(spec.provenance)
            expected_mask_hash = _sha_json(mask_record)
            if provenance.get("effect_applicability_sha256") != expected_mask_hash:
                raise MaskedMultiEndpointRuntimeError(
                    f"{spec.endpoint_id} effect applicability attestation changed"
                )
            if provenance.get("effect_applicability") != mask:
                raise MaskedMultiEndpointRuntimeError(
                    f"{spec.endpoint_id} effect applicability differs from explicit mode"
                )
            core_role = (
                "natural_language_style"
                if spec.role == "caller_natural_language"
                else spec.role
            )
            core_specs.append(
                planner.EndpointSpec(
                    endpoint_id=spec.endpoint_id,
                    name=spec.endpoint_id,
                    role=core_role,
                    normalized_parameters=embedded,
                    dsp_summary=spec.dsp_summary,
                    provenance=provenance,
                )
            )
            endpoint_modes[spec.endpoint_id] = spec.endpoint_mode
            mask_hashes[spec.endpoint_id] = expected_mask_hash
        try:
            core = planner.EndpointRegistry(renderer_mode, core_specs)
        except planner.MultiEndpointPlanError as error:
            raise MaskedMultiEndpointRuntimeError(str(error)) from error
        self.mode = renderer_mode
        self._core = core
        self._endpoint_modes = MappingProxyType(dict(endpoint_modes))
        self._mask_hashes = MappingProxyType(dict(mask_hashes))

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return self._core.endpoint_ids

    @property
    def dry_endpoint_id(self) -> str:
        return self._core.dry_endpoint_id

    def role(self, endpoint_id: str) -> str:
        return self._core.role(endpoint_id)

    def parameters(self, endpoint_id: str) -> torch.Tensor | None:
        return self._core.parameters(endpoint_id)

    def applicable(self, endpoint_id: str, effect: str) -> bool:
        if effect not in EFFECTS:
            raise MaskedMultiEndpointRuntimeError(f"unknown effect {effect!r}")
        try:
            endpoint_mode = self._endpoint_modes[endpoint_id]
        except KeyError as error:
            raise MaskedMultiEndpointRuntimeError(
                f"unknown endpoint {endpoint_id!r}"
            ) from error
        mask = (
            {"eq": False, "reverb": False}
            if endpoint_mode is None
            else effect_applicability_for_mode(endpoint_mode)
        )
        return bool(mask[effect])

    def endpoint_mode(self, endpoint_id: str) -> str | None:
        try:
            return self._endpoint_modes[endpoint_id]
        except KeyError as error:
            raise MaskedMultiEndpointRuntimeError(
                f"unknown endpoint {endpoint_id!r}"
            ) from error

    def llm_view(self) -> list[dict[str, Any]]:
        return self._core.llm_view()

    def provenance_status(self) -> dict[str, Any]:
        self.verify_effect_applicability()
        base = self._core.provenance_status()
        return {
            **base,
            "effect_applicability_verified_here": True,
            "effect_applicability": {
                endpoint_id: dict(self.mask_record(endpoint_id)["applicability"])
                for endpoint_id in self.endpoint_ids
            },
        }

    def mask_record(self, endpoint_id: str) -> dict[str, Any]:
        endpoint_mode = self.endpoint_mode(endpoint_id)
        return {
            "endpoint_id": endpoint_id,
            "renderer_mode": self.mode,
            "endpoint_mode": endpoint_mode,
            "applicability": (
                {"eq": False, "reverb": False}
                if endpoint_mode is None
                else effect_applicability_for_mode(endpoint_mode)
            ),
        }

    def verify_effect_applicability(self) -> None:
        for endpoint_id in self.endpoint_ids:
            expected = self._mask_hashes[endpoint_id]
            actual = _sha_json(self.mask_record(endpoint_id))
            if actual != expected:
                raise MaskedMultiEndpointRuntimeError(
                    f"{endpoint_id} effect applicability hash changed"
                )


class _EffectRegistryView:
    def __init__(
        self, registry: MaskedEndpointRegistry, effect: str
    ) -> None:
        self.mode = registry.mode
        self.endpoint_ids = tuple(
            endpoint_id
            for endpoint_id in registry.endpoint_ids
            if endpoint_id == registry.dry_endpoint_id
            or registry.applicable(endpoint_id, effect)
        )
        self._registry = registry

    def role(self, endpoint_id: str) -> str:
        return self._registry.role(endpoint_id)

    def parameters(self, endpoint_id: str) -> torch.Tensor | None:
        return self._registry.parameters(endpoint_id)


class MaskedMultiEndpointRenderer(renderer.MultiEndpointRenderer):
    @staticmethod
    def _effect_view(
        registry: MaskedEndpointRegistry,
        weights: torch.Tensor,
        effect: str,
    ) -> tuple[_EffectRegistryView, torch.Tensor, list[str]]:
        if not isinstance(registry, MaskedEndpointRegistry):
            raise renderer.MultiEndpointRendererError(
                "v3 renderer requires MaskedEndpointRegistryV3"
            )
        registry.verify_effect_applicability()
        view = _EffectRegistryView(registry, effect)
        indexes = [registry.endpoint_ids.index(value) for value in view.endpoint_ids]
        excluded = [
            endpoint_id
            for endpoint_id in registry.endpoint_ids
            if endpoint_id != registry.dry_endpoint_id
            and not registry.applicable(endpoint_id, effect)
        ]
        return view, weights[indexes], excluded

    def _render_multi_eq(
        self,
        audio_nct: torch.Tensor,
        registry: MaskedEndpointRegistry,
        weights: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        view, selected_weights, excluded = self._effect_view(
            registry, weights, "eq"
        )
        output, report = super()._render_multi_eq(
            audio_nct, view, selected_weights, activation
        )
        report["effect_applicability"] = {
            "effect": "eq",
            "applicable_endpoint_ids": [
                value for value in view.endpoint_ids if value != registry.dry_endpoint_id
            ],
            "excluded_endpoint_ids": excluded,
            "excluded_parameters_used": False,
        }
        return output, report

    def _render_multi_reverb(
        self,
        audio_nct: torch.Tensor,
        registry: MaskedEndpointRegistry,
        weights: torch.Tensor,
        activation: torch.Tensor,
        *,
        parameter_start: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        view, selected_weights, excluded = self._effect_view(
            registry, weights, "reverb"
        )
        output, report = super()._render_multi_reverb(
            audio_nct,
            view,
            selected_weights,
            activation,
            parameter_start=parameter_start,
        )
        report["effect_applicability"] = {
            "effect": "reverb",
            "applicable_endpoint_ids": [
                value for value in view.endpoint_ids if value != registry.dry_endpoint_id
            ],
            "excluded_endpoint_ids": excluded,
            "excluded_parameters_used": False,
        }
        return output, report


def applicability_attestation(
    *, endpoint_id: str, endpoint_mode: str | None, renderer_mode: str
) -> dict[str, Any]:
    mask = (
        {"eq": False, "reverb": False}
        if endpoint_mode is None
        else effect_applicability_for_mode(endpoint_mode)
    )
    record = {
        "endpoint_id": endpoint_id,
        "renderer_mode": renderer_mode,
        "endpoint_mode": endpoint_mode,
        "applicability": mask,
    }
    return {
        "effect_applicability": mask,
        "effect_applicability_sha256": _sha_json(record),
    }
