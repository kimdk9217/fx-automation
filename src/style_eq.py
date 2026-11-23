"""Six-band EQ settings for a signed brightness axis and rendering of a signed curve."""

from __future__ import annotations

import math

import numpy as np
import torch

import signed_curve as signed_curve_module
from automation_fx.audio import peak_report
from automation_fx.eq import (
    BAND_ORDER,
    SHELF_Q,
    BandParameters,
    SixBandParameters,
    bandwidth_q,
    neutral_six_band,
    render_six_band,
)
from automation_fx.spectrum import SpectralFeatureConfig, absolute_band_summary


def _number(value, where: str, low: float, high: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite")
    if value < low or value > high or (positive and value <= 0.0):
        raise ValueError(f"{where}={value} outside contract [{low},{high}]")
    return value


def backend_parameters(
    plan: dict,
    sample_rate: int,
    *,
    gain_range: tuple[float, float] = (-12.0, 12.0),
) -> SixBandParameters:
    gains = plan["eq_gain_db"]
    neutral = {row.band: row for row in neutral_six_band(sample_rate).bands}
    settings = {
        "low_shelf": (gains["low"], 250.0, SHELF_Q),
        "band0": (
            gains["mid"], math.sqrt(250.0 * 2000.0), bandwidth_q(250.0, 2000.0)
        ),
        "band1": (
            gains["presence"], math.sqrt(2000.0 * 6000.0), bandwidth_q(2000.0, 6000.0)
        ),
        "band2": (
            gains["high"], math.sqrt(6000.0 * 12000.0), bandwidth_q(6000.0, 12000.0)
        ),
        "band3": (0.0, float(neutral["band3"].frequency_hz), float(neutral["band3"].q_factor)),
        "high_shelf": (gains["air"], 12000.0, SHELF_Q),
    }
    return SixBandParameters(
        bands=tuple(
            BandParameters(
                band=band,
                base_gain_db=0.0,
                target_gain_db=settings[band][0],
                frequency_hz=settings[band][1],
                q_factor=settings[band][2],
            )
            for band in BAND_ORDER
        ),
        sample_rate=sample_rate,
        automation_gain_db_range=gain_range,
    )


def source_context(source: np.ndarray, sample_rate: int) -> dict:
    tensor = torch.as_tensor(source, dtype=torch.float64)
    summary = absolute_band_summary(tensor, sample_rate, SpectralFeatureConfig())
    peak = peak_report(source)
    rms = float(np.sqrt(np.mean(source**2)))
    return {
        "semantic_six_region_summary": summary,
        "peak_abs": peak["peak_abs"],
        "peak_dbfs": peak["peak_dbfs"],
        "headroom_db": -peak["peak_dbfs"] if math.isfinite(peak["peak_dbfs"]) else "+inf",
        "crest_db": 20.0 * math.log10(peak["peak_abs"] / rms) if rms > 0.0 else "+inf",
        "upstream_eq_knobs": "unknown_baked_audio; never inferred",
        "new_session_eq_state": "neutral 0 dB",
    }


PLAN_BANDS = ("low", "mid", "presence", "high", "air")
RENDER_BANDS = ("low_shelf", "band0", "band1", "band2", "high_shelf")
MAX_STRENGTH_DB = 12.0


def decode_axis(raw: dict) -> tuple[dict[str, float], dict]:
    unnormalized: dict[str, float] = {}
    for name in PLAN_BANDS:
        row = raw[name]
        direction = row["direction"]
        if direction == "unchanged":
            value = 0.0
        else:
            coordinate = _number(
                row["weight_coordinate"],
                f"eq_axis.{name}.weight_coordinate",
                -float("inf"),
                float("inf"),
            )
            magnitude = signed_curve_module.sigmoid(coordinate)
            value = -magnitude if direction == "cut" else magnitude
        unnormalized[name] = float(value)
    scale = max(abs(value) for value in unnormalized.values())
    if scale <= 1e-12:
        raise ValueError("EQ axis is zero; audible signed change is impossible")
    normalized = {name: value / scale for name, value in unnormalized.items()}
    if max(normalized.values()) - min(normalized.values()) <= 1e-6:
        raise ValueError("EQ axis is effectively broadband-only, not a spectral axis")
    return normalized, {
        "unnormalized": unnormalized,
        "normalization": "divide all five values by max absolute value",
        "normalized": normalized,
        "clamped": False,
        "preset": False,
    }


def _template(sample_rate: int) -> SixBandParameters:
    zero = {name: 0.0 for name in PLAN_BANDS}
    return backend_parameters(
        {"eq_gain_db": zero}, sample_rate, gain_range=(-MAX_STRENGTH_DB, MAX_STRENGTH_DB)
    )


def parameters_for_signed_axis(
    sample_rate: int,
    axis: dict[str, float],
    strength,
) -> SixBandParameters:
    template = _template(sample_rate)
    axis_by_band = dict(zip(RENDER_BANDS, (axis[name] for name in PLAN_BANDS)))
    rows = []
    for row in template.bands:
        value = strength * axis_by_band.get(row.band, 0.0)
        rows.append(
            BandParameters(
                band=row.band,
                base_gain_db=-value,
                target_gain_db=value,
                frequency_hz=row.frequency_hz,
                q_factor=row.q_factor,
            )
        )
    return SixBandParameters(
        tuple(rows),
        sample_rate,
        automation_gain_db_range=(-MAX_STRENGTH_DB, MAX_STRENGTH_DB),
    )


def parameters_for(axis: dict, gain_db: float, sample_rate: int,
                   band_shape: dict | None) -> SixBandParameters:
    template = parameters_for_signed_axis(sample_rate, axis, gain_db)
    if not band_shape:
        return template
    rows = []
    for row in template.bands:
        shape = band_shape.get(row.band, {})
        rows.append(BandParameters(
            band=row.band,
            base_gain_db=row.base_gain_db,
            target_gain_db=row.target_gain_db,
            frequency_hz=float(shape.get("cutoff_hz", row.frequency_hz)),
            q_factor=float(shape.get("q_factor", row.q_factor))))
    return SixBandParameters(tuple(rows), sample_rate,
                             automation_gain_db_range=template.automation_gain_db_range)


class SignedEQCLAPError(ValueError):
    pass


def render_signed_curve(
    source: torch.Tensor,
    signed_curve: torch.Tensor,
    parameters: SixBandParameters,
    *,
    backend: str = "torchlpc",
):
    if signed_curve.shape != (source.shape[0],):
        raise SignedEQCLAPError("signed curve length must match source")
    if not bool(torch.isfinite(signed_curve).all()):
        raise SignedEQCLAPError("signed curve contains NaN/Inf")
    if float(signed_curve.min()) < -1.0 or float(signed_curve.max()) > 1.0:
        raise SignedEQCLAPError("signed curve must remain inside [-1, 1]")
    fraction = (signed_curve + 1.0) * 0.5
    return render_six_band(
        source,
        parameters,
        {band: fraction for band in BAND_ORDER},
        backend=backend,
    )
