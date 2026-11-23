"""Renders audio by interpolating effect parameters between endpoints over time."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import reference_bridge as bridge
import reference_matcher as matcher
from automation_fx import eq
from automation_fx.eq import BAND_ORDER
from automation_fx.reverb import render_wet_only

DIRECTIONS = ("a_to_reference", "reference_to_a")

EFFECT_ORDER = bridge.EFFECT_ORDER


class EndpointInterpolationError(bridge.JointAutomationBridgeError):
    pass


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bridge.sha256_array(np.ascontiguousarray(value))


def _require_curve(
    value: np.ndarray | torch.Tensor,
    *,
    audio: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=audio.dtype, device=audio.device)
    expected = (audio.shape[-1],)
    if tuple(result.shape) != expected:
        raise EndpointInterpolationError(
            f"{name} must have shape {expected}, got {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise EndpointInterpolationError(f"{name} contains NaN or Inf")
    if float(result.detach().min()) < 0.0 or float(result.detach().max()) > 1.0:
        raise EndpointInterpolationError(f"{name} left [0,1]")
    return result


def _require_endpoint(
    value: torch.Tensor,
    *,
    mode: str,
    name: str,
) -> torch.Tensor:
    count = bridge.parameter_count(mode)
    result = torch.as_tensor(value)
    if tuple(result.shape) != (1, count):
        raise EndpointInterpolationError(
            f"{name} must have shape (1,{count}), got {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise EndpointInterpolationError(f"{name} contains NaN or Inf")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise EndpointInterpolationError(f"{name} left [0,1]")
    return result


def endpoint_alpha(progress: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "a_to_reference":
        return progress
    if direction == "reference_to_a":
        return 1.0 - progress
    raise EndpointInterpolationError(
        f"direction must be one of {DIRECTIONS}, got {direction!r}"
    )


def _log_lerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if bool((a <= 0.0).any()) or bool((b <= 0.0).any()):
        raise EndpointInterpolationError("log interpolation endpoints must be positive")
    return torch.exp(torch.log(a) + alpha * (torch.log(b) - torch.log(a)))


class EndpointInterpolationRenderer(bridge.JointAutomationRenderer):
    def parameter_contract(self, mode: str) -> dict[str, Any]:
        contract = super().parameter_contract(mode)
        contract.update(
            {
                "endpoint_model": "generic A and frozen reference R",
                "direction_values": list(DIRECTIONS),
                "progress_meaning": "0=directed source endpoint, 1=directed destination endpoint",
                "activation_meaning": "backend boundary envelope; 0=dry, 1=active endpoint interpolation",
                "llm_receives_dsp_parameters": False,
            }
        )
        if mode in ("eq", "eq_reverb"):
            contract["eq_interpolation"] = {
                "gain_db": "linear physical dB interpolation, then boundary activation",
                "frequency_hz": "log-space physical interpolation",
                "q_factor": "log-space physical interpolation",
                "coefficients": "recomputed per sample; no clamp or repair",
            }
        if mode in ("reverb", "eq_reverb"):
            contract["reverb_interpolation"] = {
                "method": "two fixed endpoint IR sends",
                "send_a": "activation * (1-alpha) * mix_a",
                "send_reference": "activation * alpha * mix_reference",
                "dry_gain": "1 - activation*((1-alpha)*mix_a + alpha*mix_reference)",
                "reason": "preserves causal endpoint-specific IR tails; no noncausal IR morph",
                "tail_check": (
                    "require a nonzero send, retained post-send samples, finite nonzero wet-tail "
                    "energy, and equality between the rendered post-send residual and that wet tail"
                ),
            }
        return contract

    def _render_interpolated_eq(
        self,
        audio_nct: torch.Tensor,
        endpoint_a: torch.Tensor,
        endpoint_r: torch.Tensor,
        alpha: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        params_a = self._eq_parameters(endpoint_a)
        params_r = self._eq_parameters(endpoint_r)
        audio_tc = audio_nct[0].transpose(0, 1)
        signal = audio_tc
        stages: list[dict[str, Any]] = []

        for band_a, band_r in zip(params_a.bands, params_r.bands, strict=True):
            if band_a.band != band_r.band:
                raise EndpointInterpolationError("endpoint EQ band orders differ")
            gain_a = torch.as_tensor(
                band_a.target_gain_db, dtype=signal.dtype, device=signal.device
            )
            gain_r = torch.as_tensor(
                band_r.target_gain_db, dtype=signal.dtype, device=signal.device
            )
            gain = activation * (gain_a + alpha * (gain_r - gain_a))
            frequency_a = torch.as_tensor(
                band_a.frequency_hz, dtype=signal.dtype, device=signal.device
            )
            frequency_r = torch.as_tensor(
                band_r.frequency_hz, dtype=signal.dtype, device=signal.device
            )
            q_a = torch.as_tensor(
                band_a.q_factor, dtype=signal.dtype, device=signal.device
            )
            q_r = torch.as_tensor(
                band_r.q_factor, dtype=signal.dtype, device=signal.device
            )
            frequency = _log_lerp(frequency_a, frequency_r, alpha)
            q_factor = _log_lerp(q_a, q_r, alpha)
            coefficients = eq.biquad_coefficients(
                gain,
                frequency,
                q_factor,
                self.sample_rate,
                band_a.filter_type,
            )
            coefficients["gain_db"] = gain
            stability = eq.stability_report(coefficients)
            if not stability["stable"]:
                raise EndpointInterpolationError(
                    f"{band_a.band} interpolation is unstable: "
                    f"max pole radius {stability['max_pole_radius']}, "
                    f"Jury violations {stability['jury_violations']}"
                )
            signal = eq._apply_stage(signal, coefficients, "torchlpc")
            if not bool(torch.isfinite(signal).all()):
                raise EndpointInterpolationError(
                    f"{band_a.band} produced NaN or Inf"
                )
            stages.append(
                {
                    "band": band_a.band,
                    "filter_type": band_a.filter_type,
                    "gain_db_min": float(gain.detach().min()),
                    "gain_db_max": float(gain.detach().max()),
                    "frequency_hz_min": float(frequency.detach().min()),
                    "frequency_hz_max": float(frequency.detach().max()),
                    "q_factor_min": float(q_factor.detach().min()),
                    "q_factor_max": float(q_factor.detach().max()),
                    "stability": stability,
                }
            )

        diagnostics = {
            "backend": "samplewise physical endpoint interpolation",
            "band_order": list(BAND_ORDER),
            "stages": stages,
            "gain_interpolation": "linear_db",
            "frequency_interpolation": "log_hz",
            "q_interpolation": "log_q",
            "coefficients_recomputed_per_sample": True,
            "max_pole_radius_overall": max(
                row["stability"]["max_pole_radius"] for row in stages
            ),
            "jury_violations_total": sum(
                row["stability"]["jury_violations"] for row in stages
            ),
        }
        return signal.transpose(0, 1).unsqueeze(0), diagnostics

    def _render_interpolated_reverb(
        self,
        audio_nct: torch.Tensor,
        endpoint_a: torch.Tensor,
        endpoint_r: torch.Tensor,
        alpha: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        params_a, mix_a = self._reverb_parameters(endpoint_a)
        params_r, mix_r = self._reverb_parameters(endpoint_r)
        audio_tc = audio_nct[0].transpose(0, 1)

        send_a = activation * (1.0 - alpha) * mix_a
        send_r = activation * alpha * mix_r

        noise = self._fixed_noise(params_a, audio_tc)
        wet_a = render_wet_only(
            audio_tc, send_a, params_a, seed=self.reverb_seed, noise=noise
        )
        wet_r = render_wet_only(
            audio_tc, send_r, params_r, seed=self.reverb_seed, noise=noise
        )
        effective_mix = activation * ((1.0 - alpha) * mix_a + alpha * mix_r)
        processed = (
            audio_tc * (1.0 - effective_mix).unsqueeze(1)
            + wet_a["wet_return"]
            + wet_r["wet_return"]
        )

        nonzero_send = torch.nonzero(
            (send_a.detach().abs() + send_r.detach().abs()) > 0.0
        ).reshape(-1)
        last_send_sample = int(nonzero_send[-1]) if nonzero_send.numel() else -1
        tail_start = last_send_sample + 1
        wet_tail = (
            wet_a["wet_return"][tail_start:] + wet_r["wet_return"][tail_start:]
        )
        processed_tail = processed[tail_start:]
        dry_tail = audio_tc[tail_start:]
        tail_energy = float(wet_tail.detach().square().sum())
        tail_finite = bool(torch.isfinite(wet_tail).all())
        tail_expected = last_send_sample >= 0
        required_tail_samples = (
            max(
                int(wet_a["diagnostics"]["ir_num_samples"]),
                int(wet_r["diagnostics"]["ir_num_samples"]),
            )
            - 1
            if tail_expected
            else 0
        )
        tail_context_complete = bool(
            not tail_expected or wet_tail.shape[0] >= required_tail_samples
        )
        tail_residual_matches = bool(
            not tail_expected
            or (
                wet_tail.numel()
                and torch.allclose(
                    processed_tail - dry_tail,
                    wet_tail,
                    rtol=1e-5,
                    atol=1e-8,
                )
            )
        )
        tail_preserved = bool(
            tail_context_complete
            and (not tail_expected or (tail_finite and tail_energy > 0.0))
            and tail_residual_matches
        )
        diagnostics = {
            "method": "dual_fixed_ir_send_crossfade",
            "mix_a": float(mix_a.detach()),
            "mix_reference": float(mix_r.detach()),
            "send_a_min": float(send_a.detach().min()),
            "send_a_max": float(send_a.detach().max()),
            "send_reference_min": float(send_r.detach().min()),
            "send_reference_max": float(send_r.detach().max()),
            "effective_mix_min": float(effective_mix.detach().min()),
            "effective_mix_max": float(effective_mix.detach().max()),
            "dry_gain_min": float((1.0 - effective_mix).detach().min()),
            "dry_gain_max": float((1.0 - effective_mix).detach().max()),
            "endpoint_a_ir": dict(wet_a["diagnostics"]),
            "reference_ir": dict(wet_r["diagnostics"]),
            "last_nonzero_send_sample": last_send_sample,
            "tail_expected": tail_expected,
            "required_post_send_tail_samples": required_tail_samples,
            "post_send_tail_samples": int(wet_tail.shape[0]),
            "tail_context_complete": tail_context_complete,
            "post_send_tail_energy": tail_energy,
            "post_send_tail_peak": (
                float(wet_tail.detach().abs().max()) if wet_tail.numel() else 0.0
            ),
            "post_send_tail_finite": tail_finite,
            "post_send_residual_matches_wet_tail": tail_residual_matches,
            "tail_preserved": tail_preserved,
        }
        return processed.transpose(0, 1).unsqueeze(0), diagnostics

    def render_transition(
        self,
        audio: torch.Tensor,
        endpoint_a_parameters: torch.Tensor,
        reference_parameters: torch.Tensor,
        mode: str,
        progress_curve: np.ndarray | torch.Tensor,
        *,
        direction: str,
        activation_curve: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bridge.parameter_count(mode)
        if (
            not isinstance(audio, torch.Tensor)
            or audio.ndim != 3
            or audio.shape[0] != 1
            or audio.shape[1] != 2
        ):
            raise EndpointInterpolationError(
                f"audio must have shape [1,2,N], got {getattr(audio, 'shape', None)}"
            )
        if not audio.is_floating_point() or not bool(torch.isfinite(audio).all()):
            raise EndpointInterpolationError("audio must be finite floating-point PCM")
        endpoint_a = _require_endpoint(
            endpoint_a_parameters, mode=mode, name="endpoint_a_parameters"
        ).to(dtype=audio.dtype, device=audio.device)
        endpoint_r = _require_endpoint(
            reference_parameters, mode=mode, name="reference_parameters"
        ).to(dtype=audio.dtype, device=audio.device)
        progress_before = _array_sha256(progress_curve)
        progress = _require_curve(progress_curve, audio=audio, name="progress_curve")
        if activation_curve is None:
            activation = torch.ones_like(progress)
            activation_before = _array_sha256(activation)
        else:
            activation_before = _array_sha256(activation_curve)
            activation = _require_curve(
                activation_curve, audio=audio, name="activation_curve"
            )
        alpha = endpoint_alpha(progress, direction)

        diagnostics: dict[str, Any] = {
            "mode": mode,
            "direction": direction,
            "effect_order": [],
            "progress_min": float(progress.detach().min()),
            "progress_max": float(progress.detach().max()),
            "alpha_min": float(alpha.detach().min()),
            "alpha_max": float(alpha.detach().max()),
            "activation_min": float(activation.detach().min()),
            "activation_max": float(activation.detach().max()),
        }
        output = audio
        cursor = 0
        if mode in ("eq", "eq_reverb"):
            end = matcher.EQ_PARAMETER_COUNT
            output, eq_diagnostics = self._render_interpolated_eq(
                output,
                endpoint_a[:, :end],
                endpoint_r[:, :end],
                alpha,
                activation,
            )
            diagnostics["effect_order"].append("eq")
            diagnostics["eq"] = eq_diagnostics
            cursor = end
        if mode in ("reverb", "eq_reverb"):
            end = cursor + matcher.REVERB_PARAMETER_COUNT
            output, reverb_diagnostics = self._render_interpolated_reverb(
                output,
                endpoint_a[:, cursor:end],
                endpoint_r[:, cursor:end],
                alpha,
                activation,
            )
            diagnostics["effect_order"].append("reverb")
            diagnostics["reverb"] = reverb_diagnostics

        progress_after = _array_sha256(progress_curve)
        activation_after = (
            _array_sha256(activation)
            if activation_curve is None
            else _array_sha256(activation_curve)
        )
        if progress_before != progress_after:
            raise EndpointInterpolationError("renderer mutated progress_curve")
        if activation_before != activation_after:
            raise EndpointInterpolationError("renderer mutated activation_curve")
        diagnostics.update(
            {
                "progress_sha256_before": progress_before,
                "progress_sha256_after": progress_after,
                "activation_sha256_before": activation_before,
                "activation_sha256_after": activation_after,
                "curves_unchanged": True,
                "combined_effect_order_fixed": mode != "eq_reverb"
                or diagnostics["effect_order"] == list(EFFECT_ORDER),
            }
        )
        return output, diagnostics

    def physical_endpoint_parameters(
        self,
        endpoint_a_parameters: torch.Tensor,
        reference_parameters: torch.Tensor,
        mode: str,
    ) -> dict[str, Any]:
        return {
            "endpoint_a": bridge.plain_parameters(
                self.physical_parameters(endpoint_a_parameters, mode)
            ),
            "reference": bridge.plain_parameters(
                self.physical_parameters(reference_parameters, mode)
            ),
        }
