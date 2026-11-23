"""Level, spectral and stereo summary of an audio section, given to the planner as context."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

BRIGHTNESS_CUTOFF_HZ = 3000.0


BAND_DEFINITIONS = (
    ("sub_bass", 0.0, 80.0),
    ("bass", 80.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("high", 6000.0, None),
)


class DSPSummaryError(ValueError):
    pass


@dataclass(frozen=True)
class DSPSummaryConfig:
    fft_size_samples: int = 4096
    hop_size_samples: int = 2048
    rolloff_fraction: float = 0.85
    rms_floor_dbfs: float = -120.0
    silence_peak_threshold: float = 1e-12
    minimum_sample_rate_hz: int = 16000

    def validate(self, sample_rate_hz: int) -> None:
        for name in (
            "fft_size_samples",
            "hop_size_samples",
            "minimum_sample_rate_hz",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise DSPSummaryError(f"{name} must be an integer")
        if not isinstance(sample_rate_hz, (int, np.integer)):
            raise DSPSummaryError("sample_rate_hz must be an integer")
        if int(sample_rate_hz) < self.minimum_sample_rate_hz:
            raise DSPSummaryError(
                f"sample_rate_hz must be at least {self.minimum_sample_rate_hz}"
            )
        if self.fft_size_samples < 16:
            raise DSPSummaryError("fft_size_samples must be at least 16")
        if not 0 < self.hop_size_samples <= self.fft_size_samples:
            raise DSPSummaryError(
                "hop_size_samples must be positive and no larger than the FFT size"
            )
        if not 0.0 < self.rolloff_fraction < 1.0:
            raise DSPSummaryError("rolloff_fraction must be between zero and one")
        if not math.isfinite(self.rms_floor_dbfs) or self.rms_floor_dbfs >= 0.0:
            raise DSPSummaryError("rms_floor_dbfs must be a finite negative number")
        if (
            not math.isfinite(self.silence_peak_threshold)
            or self.silence_peak_threshold <= 0.0
        ):
            raise DSPSummaryError("silence_peak_threshold must be finite and positive")

    def to_json(self, sample_rate_hz: int) -> dict[str, Any]:
        self.validate(sample_rate_hz)
        return {
            "spectral_window": "periodic_hann",
            "fft_size_samples": self.fft_size_samples,
            "hop_size_samples": self.hop_size_samples,
            "frame_pooling": (
                "sum power over channels and deterministic overlapping frames; "
                "include one final frame anchored to the clip end"
            ),
            "short_clip_padding": (
                "centered zero-padding only when shorter than one FFT frame; "
                "keeps edge impulses away from the Hann zero"
            ),
            "rolloff_fraction": self.rolloff_fraction,
            "brightness_cutoff_hz": BRIGHTNESS_CUTOFF_HZ,
            "rms_floor_dbfs": self.rms_floor_dbfs,
            "six_band_edges_hz": [
                0.0,
                80.0,
                250.0,
                500.0,
                2000.0,
                6000.0,
                float(sample_rate_hz) / 2.0,
            ],
            "normalization": "none",
        }


def _round_finite(value: float, digits: int = 6) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise DSPSummaryError("analysis produced NaN or Inf")
    rounded = round(value, digits)
    return 0.0 if rounded == 0.0 else rounded


def _as_samples_channels(audio: Any, *, name: str) -> np.ndarray:
    array = np.asarray(audio)
    if not np.issubdtype(array.dtype, np.floating):
        raise DSPSummaryError(f"{name} must contain floating-point PCM values")
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise DSPSummaryError(f"{name} must have shape [samples] or [samples, channels]")
    if array.shape[0] == 0:
        raise DSPSummaryError(f"{name} is empty")
    if array.shape[1] not in (1, 2):
        raise DSPSummaryError(f"{name} must have one or two channels in its last axis")
    if not bool(np.isfinite(array).all()):
        raise DSPSummaryError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array, dtype=np.float64)


def _frame_starts(n_samples: int, config: DSPSummaryConfig) -> list[int]:
    if n_samples <= config.fft_size_samples:
        return [0]
    last = n_samples - config.fft_size_samples
    starts = list(range(0, last + 1, config.hop_size_samples))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pooled_power(audio: np.ndarray, config: DSPSummaryConfig) -> tuple[np.ndarray, int]:
    window = np.hanning(config.fft_size_samples + 1)[:-1].astype(np.float64)
    power = np.zeros(config.fft_size_samples // 2 + 1, dtype=np.float64)
    starts = _frame_starts(audio.shape[0], config)
    for start in starts:
        frame = audio[start : start + config.fft_size_samples]
        if frame.shape[0] < config.fft_size_samples:
            padded = np.zeros((config.fft_size_samples, audio.shape[1]), dtype=np.float64)
            offset = (config.fft_size_samples - frame.shape[0]) // 2
            padded[offset : offset + frame.shape[0]] = frame
            frame = padded
        spectrum = np.fft.rfft(frame * window[:, None], axis=0)
        power += np.square(np.abs(spectrum)).sum(axis=1)
    if not bool(np.isfinite(power).all()):
        raise DSPSummaryError("spectral analysis produced NaN or Inf")
    return power, len(starts)


def _band_rows(
    power: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    nyquist_hz: float,
) -> list[dict[str, Any]]:
    total = float(power.sum())
    rows: list[dict[str, Any]] = []
    for index, (name, low_hz, declared_high_hz) in enumerate(BAND_DEFINITIONS):
        high_hz = nyquist_hz if declared_high_hz is None else declared_high_hz
        if index == len(BAND_DEFINITIONS) - 1:
            mask = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
        else:
            mask = (frequencies_hz >= low_hz) & (frequencies_hz < high_hz)
        share = 0.0 if total <= 0.0 else 100.0 * float(power[mask].sum()) / total
        rows.append(
            {
                "band": name,
                "low_hz": low_hz,
                "high_hz": _round_finite(high_hz, 3),
                "energy_share_percent": _round_finite(share),
            }
        )
    return rows


def _stereo_metrics(audio: np.ndarray, config: DSPSummaryConfig) -> dict[str, Any]:
    if audio.shape[1] != 2:
        return {
            "available": False,
            "width_side_fraction": None,
            "left_right_correlation": None,
            "note": "not reported for mono input",
        }

    if float(np.max(np.abs(audio))) <= config.silence_peak_threshold:
        return {
            "available": False,
            "width_side_fraction": None,
            "left_right_correlation": None,
            "note": "not reported for silent stereo input",
        }

    left = audio[:, 0]
    right = audio[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = math.sqrt(float(np.mean(np.square(mid))))
    side_rms = math.sqrt(float(np.mean(np.square(side))))
    width = side_rms / max(mid_rms + side_rms, config.silence_peak_threshold)

    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    correlation = (
        0.0
        if denominator <= config.silence_peak_threshold
        else float(np.dot(left_centered, right_centered)) / denominator
    )
    correlation = min(1.0, max(-1.0, correlation))
    return {
        "available": True,
        "width_side_fraction": {
            "value": _round_finite(width),
            "unit": "ratio",
            "range": [0.0, 1.0],
        },
        "left_right_correlation": {
            "value": _round_finite(correlation),
            "unit": "correlation",
            "range": [-1.0, 1.0],
        },
    }


def summarize_audio(
    audio: Any,
    sample_rate_hz: int,
    *,
    config: DSPSummaryConfig | None = None,
) -> dict[str, Any]:
    config = config or DSPSummaryConfig()
    config.validate(sample_rate_hz)
    array = _as_samples_channels(audio, name="audio")
    peak = float(np.max(np.abs(array)))
    rms = math.sqrt(float(np.mean(np.square(array))))
    is_silent = peak <= config.silence_peak_threshold
    rms_floor = 10.0 ** (config.rms_floor_dbfs / 20.0)
    rms_dbfs = 20.0 * math.log10(max(rms, rms_floor))
    crest_factor_db = (
        0.0
        if is_silent or rms <= 0.0
        else 20.0 * math.log10(max(peak / rms, 1.0))
    )

    power, frame_count = _pooled_power(array, config)
    frequencies_hz = np.fft.rfftfreq(
        config.fft_size_samples, d=1.0 / float(sample_rate_hz)
    )
    total_power = float(power.sum())
    if total_power <= config.silence_peak_threshold:
        centroid_hz = bandwidth_hz = rolloff_hz = brightness_share = 0.0
    else:
        centroid_hz = float(np.dot(frequencies_hz, power) / total_power)
        bandwidth_hz = math.sqrt(
            float(np.dot(np.square(frequencies_hz - centroid_hz), power) / total_power)
        )
        cumulative = np.cumsum(power)
        rolloff_index = int(
            np.searchsorted(cumulative, config.rolloff_fraction * total_power, side="left")
        )
        rolloff_index = min(rolloff_index, frequencies_hz.size - 1)
        rolloff_hz = float(frequencies_hz[rolloff_index])
        brightness_share = 100.0 * float(
            power[frequencies_hz >= BRIGHTNESS_CUTOFF_HZ].sum()
        ) / total_power

    return {
        "is_silent": bool(is_silent),
        "level": {
            "rms_level_dbfs": {
                "value": _round_finite(rms_dbfs),
                "unit": "dBFS",
                "meaning": "RMS loudness proxy; not LUFS",
            },
            "crest_factor_db": {
                "value": _round_finite(crest_factor_db),
                "unit": "dB",
            },
        },
        "spectrum": {
            "spectral_centroid_hz": {
                "value": _round_finite(centroid_hz, 3),
                "unit": "Hz",
            },
            "spectral_bandwidth_hz": {
                "value": _round_finite(bandwidth_hz, 3),
                "unit": "Hz",
            },
            "spectral_rolloff_85_hz": {
                "value": _round_finite(rolloff_hz, 3),
                "unit": "Hz",
            },
            "brightness_high_band_energy_share": {
                "value": _round_finite(brightness_share),
                "unit": "percent",
                "cutoff_hz": BRIGHTNESS_CUTOFF_HZ,
            },
            "six_band_energy_shares": _band_rows(
                power,
                frequencies_hz,
                nyquist_hz=float(sample_rate_hz) / 2.0,
            ),
        },
        "stereo": _stereo_metrics(array, config),
        "analysis_counts": {
            "samples": int(array.shape[0]),
            "channels": int(array.shape[1]),
            "spectral_frames": int(frame_count),
        },
    }
