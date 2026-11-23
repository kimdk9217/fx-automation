"""Time-varying reverb send (noise-shaped impulse response)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


class ReverbSendError(Exception):
    pass


def linear_convolve(signal: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    if signal.dim() != 2:
        raise ReverbSendError(f"signal must be (n_samples, n_channels): {tuple(signal.shape)}")
    if ir.dim() != 2:
        raise ReverbSendError(f"ir must be (ir_len, n_channels): {tuple(ir.shape)}")
    n, ch = signal.shape
    length, ir_ch = ir.shape
    if ir_ch != ch:
        raise ReverbSendError(f"IR channels {ir_ch} do not match signal channels {ch}")

    n_fft = 1
    while n_fft < n + length - 1:
        n_fft *= 2
    spectrum = torch.fft.rfft(signal, n=n_fft, dim=0) * torch.fft.rfft(ir, n=n_fft, dim=0)
    full = torch.fft.irfft(spectrum, n=n_fft, dim=0)
    return full[:n]


BACKEND_NAME = "dasp_compatible_noise_shaped_reverb"

N_BANDS = 12


DEFAULT_NUM_SAMPLES = 65536
DEFAULT_NUM_BANDPASS_TAPS = 1023


FIXED_MIX = 1.0


SEND_RANGE = (0.0, 1.0)


class DaspCompatibleReverbError(ReverbSendError):
    pass


def dasp_reverb_param_ranges(sample_rate: int) -> dict:
    import dasp_pytorch

    processor = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)
    return {name: tuple(bounds) for name, bounds in processor.param_ranges.items()}


@dataclass
class ReverbParameters:
    band_gains: Any
    band_decays: Any
    sample_rate: int
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_bandpass_taps: int = DEFAULT_NUM_BANDPASS_TAPS

    def __post_init__(self) -> None:
        self.band_gains = torch.as_tensor(self.band_gains, dtype=torch.float64)
        self.band_decays = torch.as_tensor(self.band_decays, dtype=torch.float64)
        self.validate()

    def validate(self) -> None:
        ranges = dasp_reverb_param_ranges(self.sample_rate)
        gain_low, gain_high = ranges["band0_gain"]
        decay_low, decay_high = ranges["band0_decay"]
        for name, values, bounds in (
            ("band_gains", self.band_gains, (gain_low, gain_high)),
            ("band_decays", self.band_decays, (decay_low, decay_high)),
        ):
            if values.shape != (N_BANDS,):
                raise DaspCompatibleReverbError(
                    f"{name} must have exactly {N_BANDS} values: {tuple(values.shape)}"
                )
            if not bool(torch.isfinite(values).all()):
                raise DaspCompatibleReverbError(f"{name} contains NaN/Inf")
            low = float(values.detach().min())
            high = float(values.detach().max())
            if low < bounds[0] or high > bounds[1]:
                raise DaspCompatibleReverbError(
                    f"{name} is outside the DASP range {list(bounds)}: [{low}, {high}]"
                )
        if self.num_bandpass_taps % 2 == 0:
            raise DaspCompatibleReverbError(
                f"num_bandpass_taps must be odd: {self.num_bandpass_taps}"
            )
        if self.num_samples < 2:
            raise DaspCompatibleReverbError(f"num_samples is too small: {self.num_samples}")

    def to_dict(self) -> dict:
        return {
            "band_gains": [float(v) for v in self.band_gains.detach()],
            "band_decays": [float(v) for v in self.band_decays.detach()],
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "num_bandpass_taps": self.num_bandpass_taps,
            "mix": FIXED_MIX,
        }

    def cache_key(self, seed: int, n_channels: int) -> str:
        payload = (
            f"{seed}|{n_channels}|{self.sample_rate}|{self.num_samples}|"
            f"{self.num_bandpass_taps}|"
            + ",".join(f"{float(v):.17g}" for v in self.band_gains.detach())
            + "|"
            + ",".join(f"{float(v):.17g}" for v in self.band_decays.detach())
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def base_noise(
    seed: int, parameters: ReverbParameters, n_channels: int = 2, dtype=torch.float64
) -> torch.Tensor:
    pad = parameters.num_bandpass_taps - 1
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        noise = torch.randn(n_channels, N_BANDS, parameters.num_samples + pad)
    return noise.to(dtype)


def noise_hash(noise: torch.Tensor) -> str:
    array = np.ascontiguousarray(noise.detach().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_ir(parameters: ReverbParameters, noise: torch.Tensor) -> torch.Tensor:
    import dasp_pytorch

    if noise.ndim != 3 or noise.shape[1] != N_BANDS:
        raise DaspCompatibleReverbError(
            f"noise must be (channels, {N_BANDS}, num_samples+pad): {tuple(noise.shape)}"
        )
    filters = dasp_pytorch.signal.octave_band_filterbank(
        parameters.num_bandpass_taps, parameters.sample_rate
    ).to(noise.dtype)
    filtered = torch.nn.functional.conv1d(noise, filters, groups=N_BANDS)
    channels = noise.shape[0]
    filtered = filtered.view(1, channels, N_BANDS, parameters.num_samples)

    t = torch.linspace(0.0, 1.0, steps=parameters.num_samples).to(noise.dtype)
    rate = (parameters.band_decays.to(noise.dtype).view(1, 1, N_BANDS, 1) * 10.0) + 1.0
    envelope = torch.exp(-rate * t.view(1, 1, 1, -1))
    shaped = filtered * envelope * parameters.band_gains.to(noise.dtype).view(1, 1, N_BANDS, 1)
    impulse = shaped.mean(2)
    return impulse[0].transpose(0, 1).contiguous()


def ir_hash(ir: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(ir.detach().numpy()).tobytes()
    ).hexdigest()


@dataclass
class IRCache:
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, parameters: ReverbParameters, seed: int, n_channels: int, dtype=torch.float64):
        key = parameters.cache_key(seed, n_channels)
        if key in self.entries:
            self.hits += 1
            return self.entries[key], True
        self.misses += 1
        noise = base_noise(seed, parameters, n_channels, dtype)
        ir = build_ir(parameters, noise)
        self.entries[key] = (noise, ir)
        return (noise, ir), False

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "n_entries": len(self.entries)}


def render_wet_only(
    dry_target: torch.Tensor,
    send_envelope: torch.Tensor,
    parameters: ReverbParameters,
    seed: int,
    cache: IRCache | None = None,
    noise: torch.Tensor | None = None,
) -> dict:
    x = torch.as_tensor(dry_target)
    if x.ndim != 2:
        raise DaspCompatibleReverbError(
            f"dry_target must be (n_samples, n_channels): {tuple(x.shape)}"
        )
    env = torch.as_tensor(send_envelope, dtype=x.dtype)
    if env.ndim != 1 or env.shape[0] != x.shape[0]:
        raise DaspCompatibleReverbError(
            f"send envelope length {tuple(env.shape)} does not match {x.shape[0]} audio samples"
        )
    if not bool(torch.isfinite(x).all()):
        raise DaspCompatibleReverbError("dry_target contains NaN/Inf")
    if not bool(torch.isfinite(env).all()):
        raise DaspCompatibleReverbError("send envelope contains NaN/Inf")
    low, high = SEND_RANGE
    if float(env.detach().min()) < low or float(env.detach().max()) > high:
        raise DaspCompatibleReverbError(
            f"send envelope must be inside {list(SEND_RANGE)}: "
            f"[{float(env.detach().min())}, {float(env.detach().max())}]"
        )

    channels = x.shape[1]
    if noise is not None:
        ir = build_ir(parameters, noise)
        used_cache = False
        used_noise = noise
    else:
        cache = cache or IRCache()
        (used_noise, ir), used_cache = cache.get(parameters, seed, channels, x.dtype)

    send_input = x * env.unsqueeze(1)
    wet = linear_convolve(send_input, ir)
    if not bool(torch.isfinite(wet).all()):
        raise DaspCompatibleReverbError("wet return contains NaN/Inf")
    return {
        "wet_return": wet,
        "send_input": send_input,
        "ir": ir,
        "noise": used_noise,
        "cache_hit": used_cache,
        "diagnostics": {
            "backend": BACKEND_NAME,
            "seed": int(seed),
            "mix": FIXED_MIX,
            "wet_only": True,
            "dry_added_by_renderer": False,
            "noise_sha256": noise_hash(used_noise),
            "ir_sha256": ir_hash(ir),
            "ir_num_samples": parameters.num_samples,
            "send_min": float(env.detach().min()),
            "send_max": float(env.detach().max()),
            "wet_energy": float((wet.detach() ** 2).sum()),
            "wet_peak_abs": float(wet.detach().abs().max()),
            "convolution": "causal linear FFT convolution",
            "global_rng_modified": False,
        },
    }
