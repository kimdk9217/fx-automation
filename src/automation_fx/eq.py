"""Six-band time-varying EQ (biquads with TorchLPC)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

GAIN_DB_RANGE = (-6.0, 6.0)


class EQError(Exception):
    pass


class MultibandEQError(EQError):
    pass


class TorchMultibandEQError(MultibandEQError):
    pass


def time_varying_fir(
    x: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise TorchMultibandEQError(f"audio must be (n_samples, n_channels): {tuple(x.shape)}")
    n, channels = x.shape
    for name, values in (("b0", b0), ("b1", b1), ("b2", b2)):
        if values.shape != (n,):
            raise TorchMultibandEQError(
                f"coefficient {name} length {tuple(values.shape)} does not match {n} samples"
            )
    if x_init is None:
        pad = torch.zeros(2, channels, dtype=x.dtype, device=x.device)
    else:
        if x_init.shape != (2, channels):
            raise TorchMultibandEQError(
                f"x_init must be (2, {channels}): {tuple(x_init.shape)}"
            )
        pad = x_init
    padded = torch.cat([pad, x], dim=0)
    return (
        b0.unsqueeze(1) * padded[2:]
        + b1.unsqueeze(1) * padded[1:-1]
        + b2.unsqueeze(1) * padded[:-2]
    )


def torchlpc_biquad(
    x: torch.Tensor,
    coefficients: dict[str, torch.Tensor],
    zi: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        from torchlpc import sample_wise_lpc
    except ImportError as exc:
        raise TorchMultibandEQError(
            "torchlpc is not installed. Install it or use backend='reference_loop'."
        ) from exc

    if x.ndim != 2:
        raise TorchMultibandEQError(f"audio must be (n_samples, n_channels): {tuple(x.shape)}")
    n, channels = x.shape
    numerator = time_varying_fir(
        x, coefficients["b0"], coefficients["b1"], coefficients["b2"], x_init
    )
    denominator = torch.stack([coefficients["a1"], coefficients["a2"]], dim=-1)
    denominator = denominator.unsqueeze(0).expand(channels, n, 2).contiguous()
    if zi is None:
        initial = torch.zeros(channels, 2, dtype=x.dtype, device=x.device)
    else:
        if zi.shape != (channels, 2):
            raise TorchMultibandEQError(
                f"zi must be ({channels}, 2): {tuple(zi.shape)}"
            )
        initial = zi
    filtered = sample_wise_lpc(
        numerator.transpose(0, 1).contiguous(), denominator, initial
    )
    return filtered.transpose(0, 1)


BAND_ORDER = ("low_shelf", "band0", "band1", "band2", "band3", "high_shelf")


BAND_FILTER_TYPE = {
    "low_shelf": "low_shelf",
    "band0": "peaking",
    "band1": "peaking",
    "band2": "peaking",
    "band3": "peaking",
    "high_shelf": "high_shelf",
}


DASP_PARAMETER_PREFIX = {
    "low_shelf": "low_shelf",
    "band0": "band0",
    "band1": "band1",
    "band2": "band2",
    "band3": "band3",
    "high_shelf": "high_shelf",
}


class DaspCompatibleEQError(MultibandEQError):
    pass


def dasp_param_ranges(sample_rate: int) -> dict:
    import dasp_pytorch

    processor = dasp_pytorch.ParametricEQ(sample_rate=sample_rate)
    return {name: tuple(bounds) for name, bounds in processor.param_ranges.items()}


def band_ranges(sample_rate: int) -> dict:
    raw = dasp_param_ranges(sample_rate)
    out = {}
    for band in BAND_ORDER:
        prefix = DASP_PARAMETER_PREFIX[band]
        out[band] = {
            "gain_db": raw[f"{prefix}_gain_db"],
            "frequency_hz": raw[f"{prefix}_cutoff_freq"],
            "q_factor": raw[f"{prefix}_q_factor"],
            "filter_type": BAND_FILTER_TYPE[band],
        }
    return out


@dataclass
class BandParameters:
    band: str
    base_gain_db: Any
    target_gain_db: Any
    frequency_hz: Any
    q_factor: Any

    def __post_init__(self) -> None:
        if self.band not in BAND_ORDER:
            raise DaspCompatibleEQError(
                f"unknown band {self.band!r}; available: {list(BAND_ORDER)}"
            )

    @property
    def filter_type(self) -> str:
        return BAND_FILTER_TYPE[self.band]

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "filter_type": self.filter_type,
            "base_gain_db": float(_scalar(self.base_gain_db)),
            "target_gain_db": float(_scalar(self.target_gain_db)),
            "frequency_hz": float(_scalar(self.frequency_hz)),
            "q_factor": float(_scalar(self.q_factor)),
        }


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0])
    return float(value)


@dataclass
class SixBandParameters:
    bands: tuple
    sample_rate: int
    automation_gain_db_range: tuple = GAIN_DB_RANGE

    def __post_init__(self) -> None:
        if len(self.bands) != len(BAND_ORDER):
            raise DaspCompatibleEQError(
                f"expected exactly {len(BAND_ORDER)} bands, got {len(self.bands)}"
            )
        names = tuple(band.band for band in self.bands)
        if names != BAND_ORDER:
            raise DaspCompatibleEQError(
                f"band order differs: {names} != {BAND_ORDER}"
            )
        self.validate()

    def validate(self) -> None:
        ranges = band_ranges(self.sample_rate)
        low, high = self.automation_gain_db_range
        for band in self.bands:
            row = ranges[band.band]
            for name, value in (
                ("frequency_hz", _scalar(band.frequency_hz)),
                ("q_factor", _scalar(band.q_factor)),
            ):
                bounds = row[name]
                if not math.isfinite(value):
                    raise DaspCompatibleEQError(f"{band.band}.{name} is not finite: {value}")
                if value < bounds[0] or value > bounds[1]:
                    raise DaspCompatibleEQError(
                        f"{band.band}.{name} {value} is outside the DASP range {list(bounds)}"
                    )
            for name, value in (
                ("base_gain_db", _scalar(band.base_gain_db)),
                ("target_gain_db", _scalar(band.target_gain_db)),
            ):
                if not math.isfinite(value):
                    raise DaspCompatibleEQError(f"{band.band}.{name} is not finite: {value}")
                if value < low or value > high:
                    raise DaspCompatibleEQError(
                        f"{band.band}.{name} {value} is outside the automation gain range "
                        f"[{low}, {high}]"
                    )
                dasp_low, dasp_high = row["gain_db"]
                if value < dasp_low or value > dasp_high:
                    raise DaspCompatibleEQError(
                        f"{band.band}.{name} {value} is also outside the DASP range "
                        f"[{dasp_low}, {dasp_high}]"
                    )

    def to_dict(self) -> dict:
        return {
            "band_order": list(BAND_ORDER),
            "bands": [band.to_dict() for band in self.bands],
            "sample_rate": self.sample_rate,
            "automation_gain_db_range": list(self.automation_gain_db_range),
        }


def neutral_six_band(sample_rate: int, frequencies=None, q_factors=None) -> SixBandParameters:
    ranges = band_ranges(sample_rate)
    bands = []
    for index, band in enumerate(BAND_ORDER):
        row = ranges[band]
        frequency = (
            frequencies[index]
            if frequencies is not None
            else math.sqrt(row["frequency_hz"][0] * row["frequency_hz"][1])
        )
        q = q_factors[index] if q_factors is not None else 1.0 / math.sqrt(2.0)
        bands.append(
            BandParameters(
                band=band,
                base_gain_db=0.0,
                target_gain_db=0.0,
                frequency_hz=frequency,
                q_factor=q,
            )
        )
    return SixBandParameters(bands=tuple(bands), sample_rate=sample_rate)


def biquad_coefficients(
    gain_db: torch.Tensor,
    frequency_hz: torch.Tensor,
    q_factor: torch.Tensor,
    sample_rate: int,
    filter_type: str,
) -> dict[str, torch.Tensor]:
    if filter_type not in ("low_shelf", "peaking", "high_shelf"):
        raise DaspCompatibleEQError(f"unsupported filter_type: {filter_type!r}")
    gain = torch.as_tensor(gain_db)
    frequency = torch.as_tensor(frequency_hz, dtype=gain.dtype, device=gain.device)
    q = torch.as_tensor(q_factor, dtype=gain.dtype, device=gain.device)
    frequency = frequency.expand_as(gain) if frequency.ndim == 0 else frequency
    q = q.expand_as(gain) if q.ndim == 0 else q

    A = torch.pow(torch.tensor(10.0, dtype=gain.dtype, device=gain.device), gain / 40.0)
    w0 = 2.0 * math.pi * (frequency / sample_rate)
    alpha = torch.sin(w0) / (2.0 * q)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)

    if bool(torch.any(a0 == 0)):
        raise DaspCompatibleEQError("a0 is zero for some samples; coefficients cannot be normalized")
    coefficients = {"b0": b0 / a0, "b1": b1 / a0, "b2": b2 / a0, "a1": a1 / a0, "a2": a2 / a0}
    for name, values in coefficients.items():
        if not bool(torch.isfinite(values).all()):
            raise DaspCompatibleEQError(f"coefficient {name} contains NaN/Inf")
    return coefficients


def gain_trajectory(band: BandParameters, curve: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(curve)
    base = torch.as_tensor(band.base_gain_db, dtype=values.dtype, device=values.device)
    target = torch.as_tensor(band.target_gain_db, dtype=values.dtype, device=values.device)
    return base + (target - base) * values


def stability_report(coefficients: dict[str, torch.Tensor]) -> dict:
    a1 = coefficients["a1"].detach()
    a2 = coefficients["a2"].detach()
    disc = a1 * a1 - 4.0 * a2
    complex_mask = disc < 0
    radius = torch.where(
        complex_mask,
        torch.sqrt(torch.abs(a2)),
        torch.maximum(
            torch.abs((-a1 + torch.sqrt(torch.clamp(disc, min=0.0))) / 2.0),
            torch.abs((-a1 - torch.sqrt(torch.clamp(disc, min=0.0))) / 2.0),
        ),
    )
    jury = (torch.abs(a2) >= 1.0) | (torch.abs(a1) >= 1.0 + a2)
    return {
        "max_pole_radius": float(torch.max(radius)),
        "jury_violations": int(torch.count_nonzero(jury)),
        "stable": bool(torch.max(radius) < 1.0 and torch.count_nonzero(jury) == 0),
        "coefficient_finite": bool(
            all(bool(torch.isfinite(v).all()) for v in coefficients.values())
        ),
        "coefficient_ranges": {
            name: {"min": float(torch.min(v.detach())), "max": float(torch.max(v.detach()))}
            for name, v in coefficients.items()
        },
    }


def _band_coefficients(
    parameters: SixBandParameters, curves: dict, n_samples: int, dtype, device
) -> dict:
    out = {}
    for band in parameters.bands:
        curve = curves.get(band.band)
        if curve is None:
            curve = torch.zeros(n_samples, dtype=dtype, device=device)
        curve = torch.as_tensor(curve, dtype=dtype, device=device)
        if curve.shape != (n_samples,):
            raise DaspCompatibleEQError(
                f"{band.band} curve length {tuple(curve.shape)} does not match {n_samples} samples"
            )
        if float(torch.min(curve)) < 0.0 or float(torch.max(curve)) > 1.0:
            raise DaspCompatibleEQError(
                f"{band.band} curve is outside [0, 1]: "
                f"[{float(torch.min(curve))}, {float(torch.max(curve))}]"
            )
        gains = gain_trajectory(band, curve)
        out[band.band] = biquad_coefficients(
            gains,
            torch.as_tensor(band.frequency_hz, dtype=dtype, device=device),
            torch.as_tensor(band.q_factor, dtype=dtype, device=device),
            parameters.sample_rate,
            band.filter_type,
        )
        out[band.band]["gain_db"] = gains
    return out


def render_six_band(
    audio: torch.Tensor,
    parameters: SixBandParameters,
    curves: dict,
    backend: str = "torchlpc",
    return_stages: bool = False,
) -> tuple[torch.Tensor, dict]:
    if backend not in ("torchlpc", "reference_loop"):
        raise DaspCompatibleEQError(f"unknown backend: {backend!r}")
    x = torch.as_tensor(audio)
    if x.ndim != 2:
        raise DaspCompatibleEQError(f"audio must be (n_samples, n_channels): {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise DaspCompatibleEQError("input audio contains NaN/Inf")
    n_samples = x.shape[0]
    coefficients = _band_coefficients(parameters, curves, n_samples, x.dtype, x.device)

    signal = x
    stages = []
    for band in BAND_ORDER:
        band_coefficients = coefficients[band]
        report = stability_report(band_coefficients)
        if not report["stable"]:
            raise DaspCompatibleEQError(
                f"{band} has unstable coefficients: max pole radius "
                f"{report['max_pole_radius']}, Jury violations in {report['jury_violations']} samples"
            )
        signal = _apply_stage(signal, band_coefficients, backend)
        if not bool(torch.isfinite(signal).all()):
            raise DaspCompatibleEQError(f"{band} output contains NaN/Inf")
        stages.append(
            {
                "band": band,
                "filter_type": BAND_FILTER_TYPE[band],
                "order_index": len(stages),
                "stability": report,
                "gain_db_min": float(torch.min(band_coefficients["gain_db"].detach())),
                "gain_db_max": float(torch.max(band_coefficients["gain_db"].detach())),
                "output": signal if return_stages else None,
            }
        )

    diagnostics = {
        "backend": backend,
        "band_order": list(BAND_ORDER),
        "n_samples": int(n_samples),
        "n_channels": int(x.shape[1]),
        "stages": [
            {k: v for k, v in stage.items() if k != "output"} for stage in stages
        ],
        "max_pole_radius_overall": max(s["stability"]["max_pole_radius"] for s in stages),
        "jury_violations_total": sum(s["stability"]["jury_violations"] for s in stages),
        "coefficients_recomputed_per_sample": True,
        "static_filter_crossfade": False,
        "input_modified_in_place": False,
    }
    if return_stages:
        diagnostics["stage_outputs"] = [stage["output"] for stage in stages]
    return signal, diagnostics


def _apply_stage(signal: torch.Tensor, coefficients: dict, backend: str) -> torch.Tensor:
    if backend == "torchlpc":
        pass

        return torchlpc_biquad(signal, coefficients)
    return reference_sample_loop(signal, coefficients)


def reference_sample_loop(
    signal: torch.Tensor, coefficients: dict[str, torch.Tensor]
) -> torch.Tensor:
    x = signal
    n, channels = x.shape
    b0, b1, b2 = coefficients["b0"], coefficients["b1"], coefficients["b2"]
    a1, a2 = coefficients["a1"], coefficients["a2"]
    outputs = []
    for channel in range(channels):
        column = x[:, channel]
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        x1 = x2 = y1 = y2 = zero
        values = []
        for index in range(n):
            xi = column[index]
            v = b0[index] * xi + b1[index] * x1 + b2[index] * x2
            yi = v - a1[index] * y1 - a2[index] * y2
            values.append(yi)
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        outputs.append(torch.stack(values))
    return torch.stack(outputs, dim=1)


SHELF_Q = 1.0 / math.sqrt(2.0)


class SemanticEQPolicyError(DaspCompatibleEQError):
    pass


def bandwidth_q(low_hz: float, high_hz: float) -> float:
    if not (math.isfinite(low_hz) and math.isfinite(high_hz) and 0.0 < low_hz < high_hz):
        raise SemanticEQPolicyError(
            f"invalid region bounds: low={low_hz}, high={high_hz}"
        )
    ratio = high_hz / low_hz
    return math.sqrt(ratio) / (ratio - 1.0)
