"""Renders EQ and reverb from Fx-Encoder++ effect parameters for endpoint audio."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

import reference_matcher as matcher
from automation_fx.eq import (
    BAND_ORDER,
    BandParameters,
    SixBandParameters,
    render_six_band,
)
from automation_fx.reverb import (
    N_BANDS,
    ReverbParameters,
    base_noise,
    dasp_reverb_param_ranges,
    render_wet_only,
)


class ReferenceAutomationBridgeError(RuntimeError):
    pass


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _linear(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return float(low) + value * (float(high) - float(low))


def _require_batch_one(audio: torch.Tensor, normalized: torch.Tensor, mode: str) -> None:
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2:
        raise ReferenceAutomationBridgeError(
            f"audio must have shape [1,2,N], got {tuple(audio.shape)}"
        )
    expected = (1, matcher.parameter_count(mode))
    if tuple(normalized.shape) != expected:
        raise ReferenceAutomationBridgeError(
            f"{mode} parameters must have shape {expected}, got {tuple(normalized.shape)}"
        )
    if not bool(torch.isfinite(normalized).all()):
        raise ReferenceAutomationBridgeError("parameters contain NaN or Inf")
    if bool(((normalized < 0.0) | (normalized > 1.0)).any()):
        raise ReferenceAutomationBridgeError("parameters left [0,1]")


class AutomationEndpointRenderer:
    def __init__(
        self,
        sample_rate: int = matcher.SAMPLE_RATE,
        *,
        reverb_seed: int = matcher.DEFAULT_REVERB_SEED,
    ) -> None:
        if sample_rate != matcher.SAMPLE_RATE:
            raise ReferenceAutomationBridgeError(
                f"Fx-Encoder++ input must be {matcher.SAMPLE_RATE} Hz"
            )
        self.sample_rate = int(sample_rate)
        self.reverb_seed = int(reverb_seed)
        official_ranges = matcher.official_eq_parameter_ranges(self.sample_rate)
        self.eq_ranges = {
            band: {
                "gain_db": official_ranges[f"{band}_gain_db"],
                "frequency_hz": official_ranges[f"{band}_cutoff_freq"],
                "q_factor": official_ranges[f"{band}_q_factor"],
            }
            for band in BAND_ORDER
        }
        self.reverb_ranges = dasp_reverb_param_ranges(self.sample_rate)
        self._noise: dict[tuple[torch.dtype, str], torch.Tensor] = {}

    def _eq_parameters(self, normalized: torch.Tensor) -> SixBandParameters:
        bands = []
        for index, band in enumerate(BAND_ORDER):
            gain_n, frequency_n, q_n = normalized[0, 3 * index : 3 * index + 3]
            ranges = self.eq_ranges[band]
            bands.append(
                BandParameters(
                    band=band,
                    base_gain_db=torch.zeros_like(gain_n),
                    target_gain_db=_linear(gain_n, *ranges["gain_db"]),
                    frequency_hz=_linear(frequency_n, *ranges["frequency_hz"]),
                    q_factor=_linear(q_n, *ranges["q_factor"]),
                )
            )
        return SixBandParameters(
            bands=tuple(bands),
            sample_rate=self.sample_rate,
            automation_gain_db_range=self.eq_ranges[BAND_ORDER[0]]["gain_db"],
        )

    def parameter_contract(self, mode: str) -> dict[str, Any]:
        matcher.parameter_count(mode)
        contract: dict[str, Any] = {
            "mode": mode,
            "normalized_optimizer_range": [0.0, 1.0],
            "mapping": "linear from sigmoid-bounded normalized values",
            "parameter_count": matcher.parameter_count(mode),
            "reverb_seed": self.reverb_seed,
        }
        if mode in ("eq", "eq_reverb"):
            contract["eq"] = {
                band: {
                    name: [float(bounds[0]), float(bounds[1])]
                    for name, bounds in ranges.items()
                }
                for band, ranges in self.eq_ranges.items()
            }
            contract["eq_range_source"] = "pinned Fx-Encoder++ official EQ processor"
        if mode in ("reverb", "eq_reverb"):
            contract["reverb"] = {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in self.reverb_ranges.items()
            }
            contract["reverb_range_source"] = (
                "official DASP Reverb normalized 12 gains, 12 decays, and mix"
            )
        return contract

    def _reverb_parameters(
        self, normalized: torch.Tensor
    ) -> tuple[ReverbParameters, torch.Tensor]:
        gain_low, gain_high = self.reverb_ranges["band0_gain"]
        decay_low, decay_high = self.reverb_ranges["band0_decay"]
        gains = _linear(normalized[0, :N_BANDS], gain_low, gain_high)
        decays = _linear(normalized[0, N_BANDS : 2 * N_BANDS], decay_low, decay_high)
        send = _linear(normalized[0, 2 * N_BANDS], *self.reverb_ranges["mix"])
        return ReverbParameters(gains, decays, self.sample_rate), send

    def _fixed_noise(self, parameters: ReverbParameters, audio: torch.Tensor) -> torch.Tensor:
        key = (audio.dtype, str(audio.device))
        if key not in self._noise:
            self._noise[key] = base_noise(
                self.reverb_seed,
                parameters,
                n_channels=2,
                dtype=audio.dtype,
            ).to(audio.device)
        return self._noise[key]

    def _render_eq(
        self, audio_nct: torch.Tensor, normalized: torch.Tensor, curve: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        params = self._eq_parameters(normalized)
        audio_tc = audio_nct[0].transpose(0, 1)
        curves = {band: curve for band in BAND_ORDER}
        out, diagnostics = render_six_band(audio_tc, params, curves)
        return out.transpose(0, 1).unsqueeze(0), diagnostics

    def _render_reverb(
        self, audio_nct: torch.Tensor, normalized: torch.Tensor, curve: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        params, full_send = self._reverb_parameters(normalized)
        audio_tc = audio_nct[0].transpose(0, 1)
        send_envelope = curve * full_send
        rendered = render_wet_only(
            audio_tc,
            send_envelope,
            params,
            seed=self.reverb_seed,
            noise=self._fixed_noise(params, audio_tc),
        )
        processed = audio_tc + rendered["wet_return"]
        nonzero_send = torch.nonzero(send_envelope.detach().abs() > 0.0).reshape(-1)
        last_send_sample = int(nonzero_send[-1]) if nonzero_send.numel() else -1
        tail_start = last_send_sample + 1
        wet_tail = rendered["wet_return"][tail_start:]
        diagnostics = {
            **rendered["diagnostics"],
            "full_send": float(full_send.detach()),
            "actual_send_min": float(send_envelope.detach().min()),
            "actual_send_max": float(send_envelope.detach().max()),
            "last_nonzero_send_sample": last_send_sample,
            "post_send_tail_samples": int(wet_tail.shape[0]),
            "post_send_tail_energy": float(wet_tail.detach().square().sum()),
            "post_send_tail_peak": (
                float(wet_tail.detach().abs().max()) if wet_tail.numel() else 0.0
            ),
        }
        return processed.transpose(0, 1).unsqueeze(0), diagnostics

    def render_with_curve(
        self,
        audio: torch.Tensor,
        normalized_parameters: torch.Tensor,
        mode: str,
        curve: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _require_batch_one(audio, normalized_parameters, mode)
        values = torch.as_tensor(curve, dtype=audio.dtype, device=audio.device)
        if values.shape != (audio.shape[-1],):
            raise ReferenceAutomationBridgeError(
                f"curve must have shape ({audio.shape[-1]},), got {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ReferenceAutomationBridgeError("curve contains NaN or Inf")
        if float(values.detach().min()) < 0.0 or float(values.detach().max()) > 1.0:
            raise ReferenceAutomationBridgeError("reference amount curve left [0,1]")

        diagnostics: dict[str, Any] = {"mode": mode, "effect_order": []}
        output = audio
        cursor = 0
        if mode in ("eq", "eq_reverb"):
            eq_values = normalized_parameters[:, : matcher.EQ_PARAMETER_COUNT]
            output, eq_diag = self._render_eq(output, eq_values, values)
            diagnostics["effect_order"].append("eq")
            diagnostics["eq"] = eq_diag
            cursor = matcher.EQ_PARAMETER_COUNT
        if mode in ("reverb", "eq_reverb"):
            reverb_values = normalized_parameters[
                :, cursor : cursor + matcher.REVERB_PARAMETER_COUNT
            ]
            output, reverb_diag = self._render_reverb(output, reverb_values, values)
            diagnostics["effect_order"].append("reverb")
            diagnostics["reverb"] = reverb_diag
        return output, diagnostics

    def render(
        self, audio: torch.Tensor, normalized_parameters: torch.Tensor, mode: str
    ) -> torch.Tensor:
        curve = torch.ones(audio.shape[-1], dtype=audio.dtype, device=audio.device)
        return self.render_with_curve(audio, normalized_parameters, mode, curve)[0]

    def physical_parameters(
        self, normalized_parameters: torch.Tensor, mode: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        count = matcher.parameter_count(mode)
        if tuple(normalized_parameters.shape) != (1, count):
            raise ReferenceAutomationBridgeError("parameter shape is invalid")
        result: dict[str, dict[str, torch.Tensor]] = {}
        cursor = 0
        if mode in ("eq", "eq_reverb"):
            params = self._eq_parameters(
                normalized_parameters[:, : matcher.EQ_PARAMETER_COUNT]
            )
            row: dict[str, torch.Tensor] = {}
            for band in params.bands:
                row[f"{band.band}_gain_db"] = torch.as_tensor(band.target_gain_db)
                row[f"{band.band}_cutoff_freq"] = torch.as_tensor(band.frequency_hz)
                row[f"{band.band}_q_factor"] = torch.as_tensor(band.q_factor)
            result["eq"] = row
            cursor = matcher.EQ_PARAMETER_COUNT
        if mode in ("reverb", "eq_reverb"):
            params, send = self._reverb_parameters(
                normalized_parameters[:, cursor : cursor + matcher.REVERB_PARAMETER_COUNT]
            )
            row = {
                **{f"band{i}_gain": value for i, value in enumerate(params.band_gains)},
                **{f"band{i}_decay": value for i, value in enumerate(params.band_decays)},
                "send": send,
            }
            result["reverb"] = row
        return result


def plain_parameters(value: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return {
        effect: {name: float(item.detach().reshape(-1)[0]) for name, item in rows.items()}
        for effect, rows in value.items()
    }


SUPPORTED_MODES = ("eq", "reverb", "eq_reverb")
PARAMETER_COUNTS = {
    "eq": matcher.EQ_PARAMETER_COUNT,
    "reverb": matcher.REVERB_PARAMETER_COUNT,
    "eq_reverb": matcher.EQ_PARAMETER_COUNT + matcher.REVERB_PARAMETER_COUNT,
}
EFFECT_ORDER = ("eq", "reverb")
STATIC_PARITY_TOLERANCES = {
    "eq": {"rtol": 1e-4, "atol": 5e-6},
    "reverb": {"rtol": 1e-5, "atol": 2e-6},
    "eq_reverb": {"rtol": 2e-4, "atol": 8e-6},
}


class JointAutomationBridgeError(ReferenceAutomationBridgeError):
    pass


def parameter_count(mode: str) -> int:
    try:
        return PARAMETER_COUNTS[mode]
    except KeyError as exc:
        raise JointAutomationBridgeError(
            f"mode must be one of {SUPPORTED_MODES}, got {mode!r}"
        ) from exc


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return sha256_array(np.ascontiguousarray(array))


class JointAutomationRenderer(AutomationEndpointRenderer):
    effect_order = EFFECT_ORDER

    def parameter_contract(self, mode: str) -> dict[str, Any]:
        parameter_count(mode)
        contract = super().parameter_contract(mode)
        contract.update(
            {
                "supported_modes": list(SUPPORTED_MODES),
                "combined_mode_supported": True,
                "combined_effect_order": list(EFFECT_ORDER),
                "official_static_parity_tolerance": dict(
                    STATIC_PARITY_TOLERANCES[mode]
                ),
                "master_curve_range": [0.0, 1.0],
            }
        )
        if mode in ("eq", "eq_reverb"):
            contract["eq_automation"] = {
                "time_varying": "band gain from 0 dB to matched gain",
                "fixed": ["cutoff frequency", "Q"],
            }
        if mode in ("reverb", "eq_reverb"):
            contract["reverb_automation"] = {
                "q": "matched_mix * master_curve",
                "wet": "convolution(q * dry, fixed IR)",
                "output": "(1 - q) * dry + wet",
                "fixed": ["band gains", "band decays"],
                "tail_after_q_returns_to_zero": True,
            }
        return contract

    def _render_reverb(
        self,
        audio_nct: torch.Tensor,
        normalized: torch.Tensor,
        curve: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        parameters, matched_mix = self._reverb_parameters(normalized)
        audio_tc = audio_nct[0].transpose(0, 1)
        q = curve * matched_mix
        rendered = render_wet_only(
            audio_tc,
            q,
            parameters,
            seed=self.reverb_seed,
            noise=self._fixed_noise(parameters, audio_tc),
        )
        wet_return = rendered["wet_return"]
        processed = audio_tc * (1.0 - q).unsqueeze(1) + wet_return

        nonzero_q = torch.nonzero(q.detach().abs() > 0.0).reshape(-1)
        last_q_sample = int(nonzero_q[-1]) if nonzero_q.numel() else -1
        tail_start = last_q_sample + 1
        wet_tail = wet_return[tail_start:]
        wet_diagnostics = dict(rendered["diagnostics"])
        wet_diagnostics["wet_renderer_internal_mix"] = wet_diagnostics.pop("mix")
        diagnostics = {
            **wet_diagnostics,
            "matched_mix": float(matched_mix.detach()),
            "q_min": float(q.detach().min()),
            "q_max": float(q.detach().max()),
            "dry_gain_min": float((1.0 - q).detach().min()),
            "dry_gain_max": float((1.0 - q).detach().max()),
            "equation": "q=matched_mix*master_curve; wet=R(q*dry); output=(1-q)*dry+wet",
            "last_nonzero_q_sample": last_q_sample,
            "post_q_tail_samples": int(wet_tail.shape[0]),
            "post_q_tail_energy": float(wet_tail.detach().square().sum()),
            "post_q_tail_peak": (
                float(wet_tail.detach().abs().max()) if wet_tail.numel() else 0.0
            ),
        }
        return processed.transpose(0, 1).unsqueeze(0), diagnostics

    def render_with_curve(
        self,
        audio: torch.Tensor,
        normalized_parameters: torch.Tensor,
        mode: str,
        curve: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        parameter_count(mode)
        before = _array_sha256(curve)
        try:
            output, diagnostics = super().render_with_curve(
                audio, normalized_parameters, mode, curve
            )
        except ReferenceAutomationBridgeError as exc:
            raise JointAutomationBridgeError(str(exc)) from exc
        after = _array_sha256(curve)
        if before != after:
            raise JointAutomationBridgeError("renderer mutated the input curve")
        diagnostics.update(
            {
                "curve_sha256_before": before,
                "curve_sha256_after": after,
                "curve_unchanged": True,
                "combined_effect_order_fixed": mode != "eq_reverb"
                or diagnostics["effect_order"] == list(EFFECT_ORDER),
            }
        )
        return output, diagnostics

    def render(
        self,
        audio: torch.Tensor,
        normalized_parameters: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        parameter_count(mode)
        curve = torch.ones(
            audio.shape[-1], dtype=audio.dtype, device=audio.device
        )
        return self.render_with_curve(
            audio, normalized_parameters, mode, curve
        )[0]

    def physical_parameters(
        self, normalized_parameters: torch.Tensor, mode: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        parameter_count(mode)
        result = super().physical_parameters(normalized_parameters, mode)
        if "reverb" in result:
            result["reverb"]["mix"] = result["reverb"].pop("send")
        return result
