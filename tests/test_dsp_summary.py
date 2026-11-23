import math

import numpy as np
import pytest

import dsp_summary as dsp

SAMPLE_RATE = 16000
CONFIG = dsp.DSPSummaryConfig(fft_size_samples=1024, hop_size_samples=512)


def _tone(frequency_hz: float, seconds: float, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.column_stack([mono, mono])


def _assert_all_json_floats_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _assert_all_json_floats_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_all_json_floats_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_seeded_noise_is_broader_than_tone_and_band_shares_sum_to_100():
    rng = np.random.default_rng(319)
    noise = rng.normal(0.0, 0.1, size=(SAMPLE_RATE * 10, 2)).astype(np.float64)
    tone = _stereo(_tone(1000.0, 10.0))
    noise_summary = dsp.summarize_audio(noise, SAMPLE_RATE, config=CONFIG)
    tone_summary = dsp.summarize_audio(tone, SAMPLE_RATE, config=CONFIG)

    assert noise_summary["spectrum"]["spectral_bandwidth_hz"]["value"] > (
        tone_summary["spectrum"]["spectral_bandwidth_hz"]["value"]
    )
    shares = noise_summary["spectrum"]["six_band_energy_shares"]
    assert sum(row["energy_share_percent"] for row in shares) == pytest.approx(
        100.0, abs=2e-5
    )


def test_stereo_width_and_correlation_distinguish_center_and_antiphase():
    mono = _tone(440.0, 1.0)
    centered = np.column_stack([mono, mono])
    antiphase = np.column_stack([mono, -mono])
    center_summary = dsp.summarize_audio(centered, SAMPLE_RATE, config=CONFIG)
    wide_summary = dsp.summarize_audio(antiphase, SAMPLE_RATE, config=CONFIG)

    assert center_summary["stereo"]["width_side_fraction"]["value"] == 0.0
    assert center_summary["stereo"]["left_right_correlation"]["value"] == 1.0
    assert wide_summary["stereo"]["width_side_fraction"]["value"] == 1.0
    assert wide_summary["stereo"]["left_right_correlation"]["value"] == -1.0


def test_short_edge_impulse_is_not_erased_by_hann_padding():
    impulse = np.zeros((100, 2), dtype=np.float64)
    impulse[0] = 0.5
    summary = dsp.summarize_audio(impulse, SAMPLE_RATE, config=CONFIG)
    shares = summary["spectrum"]["six_band_energy_shares"]
    assert summary["is_silent"] is False
    assert summary["spectrum"]["spectral_centroid_hz"]["value"] > 0.0
    assert sum(row["energy_share_percent"] for row in shares) == pytest.approx(
        100.0, abs=2e-5
    )


@pytest.mark.parametrize(
    "config,error_text",
    [
        (dsp.DSPSummaryConfig(fft_size_samples=1024.0), "fft_size_samples must be an integer"),
        (dsp.DSPSummaryConfig(hop_size_samples=512.0), "hop_size_samples must be an integer"),
    ],
)
def test_integral_config_fields_reject_float_values(config, error_text):
    with pytest.raises(dsp.DSPSummaryError, match=error_text):
        config.validate(SAMPLE_RATE)


def test_mono_reports_stereo_metrics_as_unavailable():
    summary = dsp.summarize_audio(_tone(440.0, 1.0), SAMPLE_RATE, config=CONFIG)
    assert summary["stereo"]["available"] is False
    assert summary["stereo"]["width_side_fraction"] is None
    assert summary["stereo"]["left_right_correlation"] is None
