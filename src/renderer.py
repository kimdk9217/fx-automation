"""Applies endpoint weights to the target section, sample by sample."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch

import planner as plan
import reference_matcher as matcher
from automation_fx import eq
from automation_fx.eq import BAND_ORDER
from automation_fx.reverb import render_wet_only
from endpoint_interpolation import EndpointInterpolationRenderer


class MultiEndpointRendererError(RuntimeError):
    pass


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MultiEndpointRendererError(f"{where} must be an integer")
    return int(value)


def build_boundary_activation(
    total_samples: int,
    *,
    active_start_sample: int,
    active_stop_sample: int,
    attack_start_sample: int | None = None,
    release_stop_sample: int | None = None,
) -> np.ndarray:
    total = _integer(total_samples, "total_samples")
    active_start = _integer(active_start_sample, "active_start_sample")
    active_stop = _integer(active_stop_sample, "active_stop_sample")
    attack_start = (
        active_start
        if attack_start_sample is None
        else _integer(attack_start_sample, "attack_start_sample")
    )
    release_stop = (
        active_stop
        if release_stop_sample is None
        else _integer(release_stop_sample, "release_stop_sample")
    )
    if not 0 <= attack_start <= active_start < active_stop <= release_stop <= total:
        raise MultiEndpointRendererError(
            "timing must satisfy 0 <= attack_start <= active_start < active_stop "
            "<= release_stop <= total_samples"
        )

    result = np.zeros(total, dtype=np.float64)
    result[active_start:active_stop] = 1.0
    if attack_start < active_start:
        count = active_start - attack_start
        result[attack_start : active_start + 1] = np.linspace(
            0.0, 1.0, count + 1, dtype=np.float64
        )
    if release_stop > active_stop:
        count = release_stop - active_stop
        result[active_stop:release_stop] = np.linspace(
            1.0, 0.0, count + 1, dtype=np.float64
        )[1:]
    return result


def extend_endpoint_weights(
    core_weights: np.ndarray,
    *,
    total_samples: int,
    active_start_sample: int,
    active_stop_sample: int,
) -> np.ndarray:
    core = np.asarray(core_weights, dtype=np.float64)
    if core.ndim != 2 or core.shape[1] < 1:
        raise MultiEndpointRendererError("core_weights must have shape [endpoints,samples]")
    total = _integer(total_samples, "total_samples")
    start = _integer(active_start_sample, "active_start_sample")
    stop = _integer(active_stop_sample, "active_stop_sample")
    if not 0 <= start < stop <= total or stop - start != core.shape[1]:
        raise MultiEndpointRendererError("active bounds do not match core_weights")
    result = np.empty((core.shape[0], total), dtype=np.float64)
    result[:, :start] = core[:, :1]
    result[:, start:stop] = core
    result[:, stop:] = core[:, -1:]
    return result


def _sha256(array: np.ndarray | torch.Tensor) -> str:
    if isinstance(array, torch.Tensor):
        value = array.detach().cpu().numpy()
    else:
        value = np.asarray(array)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class MultiEndpointRenderer(EndpointInterpolationRenderer):
    def _validate_audio(self, audio: torch.Tensor) -> None:
        if (
            not isinstance(audio, torch.Tensor)
            or audio.ndim != 3
            or audio.shape[0] != 1
            or audio.shape[1] != 2
            or audio.shape[-1] < 1
        ):
            raise MultiEndpointRendererError(
                f"audio must have shape [1,2,N], got {getattr(audio, 'shape', None)}"
            )
        if audio.dtype not in (torch.float32, torch.float64):
            raise MultiEndpointRendererError("audio dtype must be float32 or float64")
        if audio.device.type != "cpu":
            raise MultiEndpointRendererError(
                "version 1 renderer is CPU-only because the fixed reverb backend is CPU-only"
            )
        if not bool(torch.isfinite(audio).all()):
            raise MultiEndpointRendererError("audio must be finite floating-point PCM")

    @staticmethod
    def _runtime_weights(
        values: np.ndarray,
        *,
        audio: torch.Tensor,
        endpoint_count: int,
    ) -> torch.Tensor:
        result = torch.as_tensor(values, dtype=audio.dtype, device=audio.device)
        expected = (endpoint_count, audio.shape[-1])
        if tuple(result.shape) != expected:
            raise MultiEndpointRendererError(
                f"endpoint weights must have shape {expected}, got {tuple(result.shape)}"
            )
        if not bool(torch.isfinite(result).all()) or float(result.min()) < 0.0:
            raise MultiEndpointRendererError("endpoint weights are invalid")
        if not bool(
            torch.allclose(
                result.sum(dim=0),
                torch.ones(audio.shape[-1], dtype=audio.dtype, device=audio.device),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise MultiEndpointRendererError("endpoint weights must sum to one")
        return result

    def _non_dry_parameters(
        self,
        registry: plan.EndpointRegistry,
        *,
        start: int,
        stop: int,
        audio: torch.Tensor,
    ) -> tuple[list[str], list[torch.Tensor]]:
        endpoint_ids: list[str] = []
        parameters: list[torch.Tensor] = []
        for endpoint_id in registry.endpoint_ids:
            if registry.role(endpoint_id) == "dry":
                continue
            value = registry.parameters(endpoint_id)
            assert value is not None
            endpoint_ids.append(endpoint_id)
            parameters.append(value[:, start:stop].to(dtype=audio.dtype, device=audio.device))
        return endpoint_ids, parameters

    def _render_multi_eq(
        self,
        audio_nct: torch.Tensor,
        registry: plan.EndpointRegistry,
        weights: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        endpoint_ids, endpoint_values = self._non_dry_parameters(
            registry,
            start=0,
            stop=matcher.EQ_PARAMETER_COUNT,
            audio=audio_nct,
        )
        if not endpoint_ids:
            return audio_nct, {
                "backend": "dry identity",
                "active_endpoint_ids": [],
                "stages": [],
            }
        registry_indexes = [registry.endpoint_ids.index(endpoint_id) for endpoint_id in endpoint_ids]
        active_weights = weights[registry_indexes]
        active_weight_sum = active_weights.sum(dim=0)
        effective_weight = activation * active_weight_sum
        if float(effective_weight.detach().abs().max()) == 0.0:
            return audio_nct, {
                "backend": "dry identity",
                "active_endpoint_ids": [],
                "stages": [],
                "bit_exact_identity": True,
            }
        parameter_sets = [self._eq_parameters(value) for value in endpoint_values]
        dry_signal = audio_nct[0].transpose(0, 1)
        signal = dry_signal
        stages: list[dict[str, Any]] = []

        for band_index, band in enumerate(BAND_ORDER):
            band_rows = [parameters.bands[band_index] for parameters in parameter_sets]
            if any(row.band != band for row in band_rows):
                raise MultiEndpointRendererError("endpoint EQ band orders differ")
            gains = torch.stack(
                [torch.as_tensor(row.target_gain_db, dtype=signal.dtype, device=signal.device)
                 for row in band_rows]
            )
            frequencies = torch.stack(
                [torch.as_tensor(row.frequency_hz, dtype=signal.dtype, device=signal.device)
                 for row in band_rows]
            )
            q_factors = torch.stack(
                [torch.as_tensor(row.q_factor, dtype=signal.dtype, device=signal.device)
                 for row in band_rows]
            )
            if bool((frequencies <= 0.0).any()) or bool((q_factors <= 0.0).any()):
                raise MultiEndpointRendererError("EQ frequency and Q must be positive")

            gain = torch.sum(active_weights * gains[:, None], dim=0)
            denominator = torch.clamp(active_weight_sum, min=torch.finfo(signal.dtype).eps)
            log_frequency = torch.sum(
                active_weights * torch.log(frequencies)[:, None], dim=0
            ) / denominator
            log_q = torch.sum(
                active_weights * torch.log(q_factors)[:, None], dim=0
            ) / denominator
            dry_only = active_weight_sum <= torch.finfo(signal.dtype).eps
            log_frequency = torch.where(dry_only, torch.log(frequencies[0]), log_frequency)
            log_q = torch.where(dry_only, torch.log(q_factors[0]), log_q)
            frequency = torch.exp(log_frequency)
            q_factor = torch.exp(log_q)
            coefficients = eq.biquad_coefficients(
                gain,
                frequency,
                q_factor,
                self.sample_rate,
                band_rows[0].filter_type,
            )
            coefficients["gain_db"] = gain
            stability = eq.stability_report(coefficients)
            if not stability["stable"]:
                raise MultiEndpointRendererError(
                    f"{band} endpoint interpolation is unstable: "
                    f"max pole radius {stability['max_pole_radius']}, "
                    f"Jury violations {stability['jury_violations']}"
                )
            signal = eq._apply_stage(signal, coefficients, "torchlpc")
            if not bool(torch.isfinite(signal).all()):
                raise MultiEndpointRendererError(f"{band} produced NaN or Inf")
            stages.append(
                {
                    "band": band,
                    "filter_type": band_rows[0].filter_type,
                    "gain_db_min": float(gain.detach().min()),
                    "gain_db_max": float(gain.detach().max()),
                    "frequency_hz_min": float(frequency.detach().min()),
                    "frequency_hz_max": float(frequency.detach().max()),
                    "q_factor_min": float(q_factor.detach().min()),
                    "q_factor_max": float(q_factor.detach().max()),
                    "stability": stability,
                }
            )
        signal = dry_signal + activation.unsqueeze(1) * (signal - dry_signal)
        return signal.transpose(0, 1).unsqueeze(0), {
            "backend": "samplewise weighted physical endpoint interpolation",
            "active_endpoint_ids": endpoint_ids,
            "dry_endpoint_gain_db": 0.0,
            "gain_interpolation": "linear dB including dry as zero gain",
            "boundary_activation": "output crossfade from dry to planned EQ output",
            "frequency_interpolation": "normalized log-space over non-dry endpoints",
            "q_interpolation": "normalized log-space over non-dry endpoints",
            "coefficients_recomputed_per_sample": True,
            "stages": stages,
            "max_pole_radius_overall": max(
                row["stability"]["max_pole_radius"] for row in stages
            ),
            "jury_violations_total": sum(
                row["stability"]["jury_violations"] for row in stages
            ),
        }

    def _render_multi_reverb(
        self,
        audio_nct: torch.Tensor,
        registry: plan.EndpointRegistry,
        weights: torch.Tensor,
        activation: torch.Tensor,
        *,
        parameter_start: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        endpoint_ids, endpoint_values = self._non_dry_parameters(
            registry,
            start=parameter_start,
            stop=parameter_start + matcher.REVERB_PARAMETER_COUNT,
            audio=audio_nct,
        )
        audio_tc = audio_nct[0].transpose(0, 1)
        registry_indexes = [registry.endpoint_ids.index(endpoint_id) for endpoint_id in endpoint_ids]
        wet_returns: list[torch.Tensor] = []
        endpoint_reports: list[dict[str, Any]] = []
        sends: list[torch.Tensor] = []
        injection_magnitudes: list[torch.Tensor] = []
        required_tail_samples = 0

        for endpoint_id, normalized, registry_index in zip(
            endpoint_ids, endpoint_values, registry_indexes, strict=True
        ):
            parameters, mix = self._reverb_parameters(normalized)
            send = activation * weights[registry_index] * mix
            sends.append(send)
            send_input = audio_tc * send.unsqueeze(1)
            injection_magnitude = send_input.detach().abs().amax(dim=1)
            injection_magnitudes.append(injection_magnitude)
            has_injection = bool((injection_magnitude > 0.0).any())
            if float(send.detach().abs().max()) == 0.0:
                wet_return = torch.zeros_like(audio_tc)
                rendered_diagnostics: dict[str, Any] = {
                    "skipped_zero_send": True,
                    "ir_num_samples": 0,
                }
            else:
                rendered = render_wet_only(
                    audio_tc,
                    send,
                    parameters,
                    seed=self.reverb_seed,
                    noise=self._fixed_noise(parameters, audio_tc),
                )
                wet_return = rendered["wet_return"]
                rendered_diagnostics = dict(rendered["diagnostics"])
                if has_injection:
                    required_tail_samples = max(
                        required_tail_samples,
                        int(rendered_diagnostics["ir_num_samples"]) - 1,
                    )
            wet_returns.append(wet_return)
            endpoint_reports.append(
                {
                    "endpoint_id": endpoint_id,
                    "send_min": float(send.detach().min()),
                    "send_max": float(send.detach().max()),
                    "send_input_peak": float(send_input.detach().abs().max()),
                    "actual_audio_injection": has_injection,
                    "ir": rendered_diagnostics,
                }
            )

        if sends:
            total_send = torch.stack(sends).sum(dim=0)
            total_wet = torch.stack(wet_returns).sum(dim=0)
        else:
            total_send = torch.zeros(audio_tc.shape[0], dtype=audio_tc.dtype, device=audio_tc.device)
            total_wet = torch.zeros_like(audio_tc)
        if float(total_send.detach().max()) > 1.0 + 1e-6:
            raise MultiEndpointRendererError("summed reverb send exceeded one")
        processed = audio_tc * (1.0 - total_send).unsqueeze(1) + total_wet

        if injection_magnitudes:
            total_injection = torch.stack(injection_magnitudes).amax(dim=0)
        else:
            total_injection = torch.zeros(
                audio_tc.shape[0], dtype=audio_tc.dtype, device=audio_tc.device
            )
        nonzero_injection = torch.nonzero(total_injection > 0.0).reshape(-1)
        last_injection = (
            int(nonzero_injection[-1]) if nonzero_injection.numel() else -1
        )
        tail_start = last_injection + 1
        wet_tail = total_wet[tail_start:]
        dry_tail = audio_tc[tail_start:]
        processed_tail = processed[tail_start:]
        tail_expected = last_injection >= 0
        tail_context_complete = bool(
            not tail_expected or wet_tail.shape[0] >= required_tail_samples
        )
        residual = processed_tail - dry_tail
        residual_error = residual - wet_tail
        residual_max_abs_error = (
            float(residual_error.detach().abs().max())
            if residual_error.numel()
            else 0.0
        )
        residual_scale = max(
            float(processed_tail.detach().abs().max()) if processed_tail.numel() else 0.0,
            float(dry_tail.detach().abs().max()) if dry_tail.numel() else 0.0,
            float(wet_tail.detach().abs().max()) if wet_tail.numel() else 0.0,
            float(torch.finfo(audio_tc.dtype).tiny),
        )
        residual_atol = 4.0 * float(torch.finfo(audio_tc.dtype).eps) * residual_scale
        residual_rtol = 1e-5
        residual_matches = bool(
            not tail_expected
            or (
                wet_tail.numel()
                and torch.allclose(
                    residual,
                    wet_tail,
                    rtol=residual_rtol,
                    atol=residual_atol,
                )
            )
        )
        tail_energy = float(wet_tail.detach().square().sum())
        tail_preserved = bool(
            tail_context_complete
            and (not tail_expected or bool(torch.isfinite(wet_tail).all()))
            and residual_matches
        )
        return processed.transpose(0, 1).unsqueeze(0), {
            "method": "fixed_endpoint_ir_send_bank",
            "true_ir_interpolation": False,
            "active_endpoint_ids": endpoint_ids,
            "dry_endpoint_send": 0.0,
            "endpoint_reports": endpoint_reports,
            "total_send_min": float(total_send.detach().min()),
            "total_send_max": float(total_send.detach().max()),
            "dry_gain_min": float((1.0 - total_send).detach().min()),
            "dry_gain_max": float((1.0 - total_send).detach().max()),
            "last_nonzero_audio_injection_sample": last_injection,
            "tail_expected": tail_expected,
            "required_post_send_tail_samples": required_tail_samples,
            "post_send_tail_samples": int(wet_tail.shape[0]),
            "tail_context_complete": tail_context_complete,
            "post_send_tail_energy": tail_energy,
            "post_send_tail_nonzero": bool(tail_energy > 0.0),
            "post_send_tail_peak": (
                float(wet_tail.detach().abs().max()) if wet_tail.numel() else 0.0
            ),
            "post_send_residual_matches_wet_tail": residual_matches,
            "post_send_residual_max_abs_error": residual_max_abs_error,
            "post_send_residual_comparison_atol": residual_atol,
            "post_send_residual_comparison_rtol": residual_rtol,
            "tail_preserved": tail_preserved,
            "limitation": (
                "crossfades sends to fixed endpoint IRs; simultaneous rooms may be heard "
                "during transitions and cost grows with non-dry endpoint count"
            ),
        }

    def render_plan(
        self,
        audio: torch.Tensor,
        *,
        registry: plan.EndpointRegistry,
        validated_plan: Mapping[str, Any],
        plan_schema: Mapping[str, Any],
        instruction: str,
        active_start_sample: int,
        active_stop_sample: int,
        attack_start_sample: int | None = None,
        release_stop_sample: int | None = None,
        require_complete_reverb_tail: bool = True,
        require_safe_output: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._validate_audio(audio)
        if not isinstance(validated_plan, Mapping):
            raise MultiEndpointRendererError("validated_plan must be an object")
        if not isinstance(plan_schema, Mapping):
            raise MultiEndpointRendererError("plan_schema must be the externally locked schema")
        if not isinstance(instruction, str) or not instruction.strip():
            raise MultiEndpointRendererError("instruction must be the original non-empty text")
        try:
            clean_plan = plan.validate_plan(
                validated_plan,
                schema=plan_schema,
                registry=registry,
                instruction=instruction,
            )
        except plan.MultiEndpointPlanError as error:
            raise MultiEndpointRendererError(
                f"renderer rejected unvalidated plan: {error}"
            ) from error
        if registry.mode != clean_plan.get("effect_mode"):
            raise MultiEndpointRendererError("plan and endpoint registry modes differ")
        if clean_plan.get("schema_version") != plan.SCHEMA_VERSION:
            raise MultiEndpointRendererError("plan schema version is invalid")
        start = _integer(active_start_sample, "active_start_sample")
        stop = _integer(active_stop_sample, "active_stop_sample")
        if not 0 <= start < stop <= audio.shape[-1]:
            raise MultiEndpointRendererError("active bounds are outside audio")
        if attack_start_sample is None or release_stop_sample is None:
            raise MultiEndpointRendererError(
                "explicit attack_start_sample and release_stop_sample are required; "
                "one-sample default boundary jumps are forbidden"
            )
        attack_start = _integer(attack_start_sample, "attack_start_sample")
        release_stop = _integer(release_stop_sample, "release_stop_sample")
        minimum_boundary_samples = max(2, int(np.ceil(0.010 * self.sample_rate)))
        if (
            start - attack_start < minimum_boundary_samples
            or release_stop - stop < minimum_boundary_samples
        ):
            raise MultiEndpointRendererError(
                "boundary attack and release each need at least 10 ms of sample intervals"
            )

        core_weights, schedule_report = plan.evaluate_endpoint_schedule(
            clean_plan,
            registry=registry,
            n_samples=stop - start,
        )
        weights_array = extend_endpoint_weights(
            core_weights,
            total_samples=audio.shape[-1],
            active_start_sample=start,
            active_stop_sample=stop,
        )
        activation_array = build_boundary_activation(
            audio.shape[-1],
            active_start_sample=start,
            active_stop_sample=stop,
            attack_start_sample=attack_start_sample,
            release_stop_sample=release_stop_sample,
        )
        weights_before = _sha256(weights_array)
        activation_before = _sha256(activation_array)
        weights = self._runtime_weights(
            weights_array,
            audio=audio,
            endpoint_count=len(registry.endpoint_ids),
        )
        activation = torch.as_tensor(
            activation_array, dtype=audio.dtype, device=audio.device
        )

        output = audio
        diagnostics: dict[str, Any] = {
            "mode": registry.mode,
            "effect_order": [],
            "schedule": schedule_report,
            "boundary_activation": {
                "owned_by": "deterministic backend",
                "active_start_sample": start,
                "active_stop_sample_exclusive": stop,
                "attack_start_sample": attack_start,
                "release_stop_sample_exclusive": release_stop,
                "minimum_ramp_samples": minimum_boundary_samples,
                "minimum_ramp_seconds": minimum_boundary_samples / self.sample_rate,
                "minimum": float(activation.min()),
                "maximum": float(activation.max()),
            },
            "endpoint_provenance": registry.provenance_status(),
        }
        parameter_start = 0
        if registry.mode in ("eq", "eq_reverb"):
            output, eq_report = self._render_multi_eq(
                output, registry, weights, activation
            )
            diagnostics["effect_order"].append("eq")
            diagnostics["eq"] = eq_report
            parameter_start = matcher.EQ_PARAMETER_COUNT
        if registry.mode in ("reverb", "eq_reverb"):
            output, reverb_report = self._render_multi_reverb(
                output,
                registry,
                weights,
                activation,
                parameter_start=parameter_start,
            )
            diagnostics["effect_order"].append("reverb")
            diagnostics["reverb"] = reverb_report
            if require_complete_reverb_tail and reverb_report["tail_expected"]:
                failed_tail_checks = [
                    name
                    for name in (
                        "tail_context_complete",
                        "post_send_residual_matches_wet_tail",
                        "tail_preserved",
                    )
                    if reverb_report[name] is not True
                ]
                if failed_tail_checks:
                    raise MultiEndpointRendererError(
                        "reverb tail verification failed: "
                        + ", ".join(failed_tail_checks)
                    )

        if not bool(torch.isfinite(output).all()):
            raise MultiEndpointRendererError("renderer produced NaN or Inf")
        input_peak = float(audio.detach().abs().max())
        output_peak = float(output.detach().abs().max())
        input_clipping_samples = int((audio.detach().abs() > 1.0).sum())
        output_clipping_samples = int((output.detach().abs() > 1.0).sum())
        diagnostics["safety"] = {
            "input_peak_abs": input_peak,
            "output_peak_abs": output_peak,
            "input_samples_over_1_0": input_clipping_samples,
            "output_samples_over_1_0": output_clipping_samples,
            "output_clipping_ratio": output_clipping_samples / output.numel(),
            "limiter": False,
            "trim_or_normalization": False,
        }
        if require_safe_output and output_clipping_samples:
            raise MultiEndpointRendererError(
                "raw output safety failed: "
                f"{output_clipping_samples} samples exceed absolute 1.0 "
                f"(peak={output_peak})"
            )
        weights_after = _sha256(weights_array)
        activation_after = _sha256(activation_array)
        if weights_before != weights_after or activation_before != activation_after:
            raise MultiEndpointRendererError("renderer mutated resolved control curves")
        diagnostics.update(
            {
                "endpoint_weights_sha256_before": weights_before,
                "endpoint_weights_sha256_after": weights_after,
                "activation_sha256_before": activation_before,
                "activation_sha256_after": activation_after,
                "control_curves_unchanged": True,
                "planner_payload_visibility": "not observable inside renderer",
                "renderer_optimizer_steps": 0,
            }
        )
        return output, diagnostics
