"""SocialFX word EQ curves and the features used to compare rendered audio with them."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import sosfilt

EQ_FREQS = (20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843, 971,
            1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462, 3901, 4392,
            4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474, 13984, 15675,
            17566, 19682)

EQ_Q = 4.31
N_BANDS = len(EQ_FREQS)


class AudealizeError(ValueError):
    pass


def curve_to_gains_db(curve, gain_range: float = 1.0) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (N_BANDS,):
        raise AudealizeError(f"curve must have {N_BANDS} values: {values.shape}")
    if not np.isfinite(values).all():
        raise AudealizeError("curve contains NaN/Inf")
    lo, hi = float(values.min()), float(values.max())
    norm = np.zeros_like(values) if hi == lo else ((values - lo) / (hi - lo)) * 2.0 - 1.0
    return float(gain_range) * 5.0 * norm


def _peaking_sos(freq_hz: float, q: float, gain_db: float, sample_rate: int):
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * float(freq_hz) / float(sample_rate)
    alpha = np.sin(w0) / (2.0 * float(q))
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def sos_for_curve(curve, sample_rate: int, gain_range: float = 1.0) -> np.ndarray:
    gains = curve_to_gains_db(curve, gain_range)
    return np.array([_peaking_sos(f, EQ_Q, g, sample_rate)
                     for f, g in zip(EQ_FREQS, gains)], dtype=np.float64)


def apply_eq(audio: np.ndarray, curve, sample_rate: int,
             gain_range: float = 1.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise AudealizeError(f"audio must be (n,) or (n, ch): {x.shape}")
    sos = sos_for_curve(curve, sample_rate, gain_range)
    return np.stack([sosfilt(sos, x[:, c]) for c in range(x.shape[1])], axis=1)


FEATURE_NAMES = [
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    *[f"spectral_contrast_{i}" for i in range(7)],
    *[f"mfcc_{i}" for i in range(13)],
    "rms_mean", "rms_std", "zcr_mean", "zcr_std",
    "crest_factor", "brightness", "loudness_db",
]
FEATURE_SR = 22050


def extract_features(audio: np.ndarray, sr: int = FEATURE_SR) -> np.ndarray:
    import librosa

    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1) if y.shape[1] <= y.shape[0] else y.mean(axis=0)
    if float(np.max(np.abs(y))) < 1e-8:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    f = []
    for fn in (librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff):
        v = fn(y=y, sr=sr)[0]
        f += [float(np.mean(v)), float(np.std(v))]
    v = librosa.feature.spectral_flatness(y=y)[0]
    f += [float(np.mean(v)), float(np.std(v))]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f += [float(np.mean(contrast[i])) for i in range(contrast.shape[0])]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f += [float(np.mean(mfcc[i])) for i in range(13)]
    rms = librosa.feature.rms(y=y)[0]
    f += [float(np.mean(rms)), float(np.std(rms))]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f += [float(np.mean(zcr)), float(np.std(zcr))]
    rms_val = float(np.sqrt(np.mean(y ** 2)))
    f.append(float(np.max(np.abs(y)) / (rms_val + 1e-8)))
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    f.append(float(np.sum(S[freqs >= 1500, :]) / (np.sum(S) + 1e-8)))
    f.append(20.0 * float(np.log10(rms_val + 1e-8)))
    return np.asarray(f, dtype=np.float32)


def standardized_distance(x: np.ndarray, reference: np.ndarray) -> float:
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-8
    z = (np.asarray(x, dtype=np.float64).reshape(-1) - mu) / sd
    return float(np.sqrt(np.mean(z ** 2)))


MAX_CURVES = 20


def equal_windows(n_samples: int, k: int):
    edges = np.linspace(0, n_samples, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b > a]


def _features(mono, a, b, sample_rate):
    import librosa
    seg = librosa.resample(np.asarray(mono[a:b], dtype=np.float64),
                           orig_sr=sample_rate, target_sr=FEATURE_SR)
    return extract_features(seg, FEATURE_SR)


def build_reference(sources, curves_by_word, sample_rate, k):
    ref = defaultdict(list)
    for word, curves in curves_by_word.items():
        for curve in curves[:MAX_CURVES]:
            for src in sources:
                mono = src.mean(axis=1)
                wet = apply_eq(mono, curve, sample_rate)[:, 0]
                ref[word].append([_features(wet, a, b, sample_rate)
                                  for a, b in equal_windows(len(mono), k)])
    return {w: np.array(v) for w, v in ref.items()}
