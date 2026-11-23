"""Band energies and spectral centroid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from automation_fx.bezier import REGION_HZ

REGION_ORDER = tuple(REGION_HZ)
SEMANTIC_REGION_BAND_EDGES_HZ = tuple(
    float(REGION_HZ[name][0]) for name in REGION_ORDER
)
STAGE30_BAND_EDGES_HZ = (0.0, 250.0, 2000.0, 6000.0, 12000.0)
BAND_EDGES_HZ = SEMANTIC_REGION_BAND_EDGES_HZ

FEATURE_NAMES = tuple(f"band_{name}" for name in REGION_ORDER) + (
    "rms_log_energy_change",
)
STAGE30_FEATURE_NAMES = (
    "band_0_250_hz",
    "band_250_2000_hz",
    "band_2000_6000_hz",
    "band_6000_12000_hz",
    "band_12000_nyquist_hz",
    "rms_log_energy_change",
)


class DSPFeatureError(Exception):
    pass


@dataclass(frozen=True)
class SpectralFeatureConfig:
    n_fft: int = 2048
    hop_length: int = 512
    eps: float = 1e-12
    band_edges_hz: tuple[float, ...] = BAND_EDGES_HZ
    center: bool = False

    def validate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise DSPFeatureError(f"invalid sample_rate: {sample_rate}")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise DSPFeatureError(f"invalid n_fft/hop_length: {self.n_fft}, {self.hop_length}")
        if self.hop_length > self.n_fft:
            raise DSPFeatureError(
                f"hop_length {self.hop_length} is larger than n_fft {self.n_fft}"
            )
        if self.eps <= 0.0:
            raise DSPFeatureError(f"eps must be positive: {self.eps}")
        nyquist = sample_rate / 2.0
        edges = self.band_edges_hz
        if len(edges) < 2:
            raise DSPFeatureError(f"too few band edges: {edges}")
        if not all(isinstance(value, (int, float)) for value in edges):
            raise DSPFeatureError(f"band edges must be numbers: {edges}")
        if edges[0] < 0.0:
            raise DSPFeatureError(f"the first band edge must be >= 0: {edges[0]}")
        for low, high in zip(edges, edges[1:]):
            if not low < high:
                raise DSPFeatureError(f"band edges must be strictly increasing: {edges}")
        if edges[-1] >= nyquist:
            raise DSPFeatureError(
                f"the last band edge {edges[-1]} Hz is at or above Nyquist {nyquist} Hz"
            )

    @property
    def n_bands(self) -> int:
        return len(self.band_edges_hz)

    def to_dict(self, sample_rate: int) -> dict:
        self.validate(sample_rate)
        edges = list(self.band_edges_hz) + [sample_rate / 2.0]
        if self.band_edges_hz == SEMANTIC_REGION_BAND_EDGES_HZ:
            names = list(FEATURE_NAMES)
            region_order = list(REGION_ORDER)
            profile = "semantic-regions-1"
        elif self.band_edges_hz == STAGE30_BAND_EDGES_HZ:
            names = list(STAGE30_FEATURE_NAMES)
            region_order = None
            profile = "stage30-five-band-1"
        else:
            names = [
                f"band_{low:g}_{high:g}_hz"
                for low, high in zip(edges[:-2], edges[1:-1])
            ] + [f"band_{edges[-2]:g}_nyquist_hz", "rms_log_energy_change"]
            region_order = None
            profile = "custom"
        return {
            "profile": profile,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "center": self.center,
            "window": "hann",
            "eps": self.eps,
            "band_edges_hz": edges,
            "feature_names": names,
            "semantic_region_order": region_order,
            "excluded_dc_subsonic_hz": (
                [0.0, float(self.band_edges_hz[0])]
                if self.band_edges_hz[0] > 0.0
                else None
            ),
            "relative_to_source": True,
            "normalization": "none",
            "same_settings_for_processed_and_source": True,
            "frame_vs_downbeat": (
                "Per-bar features use center=False STFT on pieces cut at downbeat samples; "
                "no frame crosses a bar boundary."
            ),
        }


def _stft_power(audio: torch.Tensor, config: SpectralFeatureConfig) -> torch.Tensor:
    if audio.ndim != 2:
        raise DSPFeatureError(f"audio must be (n_samples, n_channels): {tuple(audio.shape)}")
    n_samples = audio.shape[0]
    if n_samples < config.n_fft:
        raise DSPFeatureError(
            f"piece length {n_samples} is shorter than n_fft {config.n_fft}"
        )
    window = torch.hann_window(config.n_fft, dtype=audio.dtype, device=audio.device)
    spectrum = torch.stft(
        audio.transpose(0, 1),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    return torch.sum(torch.abs(spectrum) ** 2, dim=(0, 2))


def band_masks(sample_rate: int, config: SpectralFeatureConfig, device) -> list[torch.Tensor]:
    config.validate(sample_rate)
    freqs = torch.fft.rfftfreq(config.n_fft, d=1.0 / sample_rate).to(device)
    edges = list(config.band_edges_hz) + [float(sample_rate) / 2.0 + 1.0]
    masks = []
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        if index == len(edges) - 2:
            masks.append(freqs >= low)
        else:
            masks.append((freqs >= low) & (freqs < high))
    return masks


def band_energies(
    audio: torch.Tensor, sample_rate: int, config: SpectralFeatureConfig
) -> torch.Tensor:
    power = _stft_power(audio, config)
    masks = band_masks(sample_rate, config, audio.device)
    return torch.stack([torch.sum(power[mask]) for mask in masks])


def spectral_centroid_hz(
    audio: torch.Tensor, sample_rate: int, config: SpectralFeatureConfig
) -> torch.Tensor:
    power = _stft_power(audio, config)
    magnitude = torch.sqrt(power + config.eps)
    freqs = torch.fft.rfftfreq(config.n_fft, d=1.0 / sample_rate).to(audio.device)
    return torch.sum(freqs * magnitude) / (torch.sum(magnitude) + config.eps)


def absolute_band_summary(
    audio: torch.Tensor, sample_rate: int, config: SpectralFeatureConfig
) -> dict:
    energies = band_energies(audio, sample_rate, config)
    total = torch.sum(energies)
    share_db = 10.0 * torch.log10((energies + config.eps) / (total + config.eps))
    rms = torch.sqrt(torch.mean(audio**2))
    edges = list(config.band_edges_hz) + [float(sample_rate) / 2.0]
    names = config.to_dict(sample_rate)["feature_names"][:-1]
    with torch.no_grad():
        return {
            "band_names": list(names),
            "band_edges_hz": edges,
            "band_energy": [float(v) for v in energies],
            "band_share_db": [round(float(v), 3) for v in share_db],
            "rms": float(rms),
            "rms_dbfs": (
                round(float(20.0 * torch.log10(rms + config.eps)), 3)
                if float(rms) > 0.0
                else "-inf"
            ),
            "spectral_centroid_hz_auxiliary": round(
                float(spectral_centroid_hz(audio, sample_rate, config)), 1
            ),
            "n_samples": int(audio.shape[0]),
            "duration_sec": round(audio.shape[0] / sample_rate, 4),
            "note": "section summary, not per-frame values; the centroid is auxiliary.",
        }
