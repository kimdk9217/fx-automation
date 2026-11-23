"""Fits EQ and reverb parameters to a reference clip with a frozen Fx-Encoder++."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from automation_fx.paths import FXENCODER_REPO

OFFICIAL_REPO = Path(
    os.environ.get(
        "FXENCODER_OFFICIAL_REPO",
        str(FXENCODER_REPO),
    )
)
OFFICIAL_COMMIT = "7b1e1f7efaa8a7c212ae4d62c01a5bac5f61c47b"
SAMPLE_RATE = 44_100
MATCH_STEPS = 60
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_REVERB_SEED = 41_004

EQ_PARAMETER_COUNT = 18
REVERB_PARAMETER_COUNT = 25
PARAM_COUNTS: Mapping[str, int] = {
    "eq": EQ_PARAMETER_COUNT,
    "reverb": REVERB_PARAMETER_COUNT,
    "eq_reverb": EQ_PARAMETER_COUNT + REVERB_PARAMETER_COUNT,
}
MODES = tuple(PARAM_COUNTS)


class ReferenceFxMatcherError(RuntimeError):
    pass


def _load_official_processors():
    if not (OFFICIAL_REPO / "fx_chain" / "fx_processors.py").is_file():
        raise ReferenceFxMatcherError(f"Fx-Encoder++ checkout missing: {OFFICIAL_REPO}")

    repo_text = str(OFFICIAL_REPO)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    shimmed_torchcomp = False
    if importlib.util.find_spec("torchcomp") is None and "torchcomp" not in sys.modules:
        sys.modules["torchcomp"] = types.ModuleType("torchcomp")
        shimmed_torchcomp = True

    try:
        import dasp_pytorch
        from fx_aug import select_processors
        from fx_chain.fx_processors import Reverb
    except Exception as exc:
        raise ReferenceFxMatcherError(f"official processor import failed: {exc}") from exc
    finally:
        if shimmed_torchcomp:
            sys.modules.pop("torchcomp", None)

    reverb_module = sys.modules.get(Reverb.__module__)
    reverb_source = Path(getattr(reverb_module, "__file__", "")).resolve()
    if OFFICIAL_REPO.resolve() not in reverb_source.parents:
        raise ReferenceFxMatcherError(
            f"Reverb did not load from official checkout: {reverb_source}"
        )
    return dasp_pytorch.ParametricEQ, Reverb, select_processors, shimmed_torchcomp


(
    _ParametricEQ,
    _OfficialReverb,
    _select_official_processor,
    TORCHCOMP_IMPORT_SHIMMED,
) = _load_official_processors()


def parameter_count(mode: str) -> int:
    try:
        return PARAM_COUNTS[mode]
    except KeyError as exc:
        raise ReferenceFxMatcherError(f"mode must be one of {MODES}, got {mode!r}") from exc


def official_eq_parameter_ranges(
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, tuple[float, float]]:
    if sample_rate != SAMPLE_RATE:
        raise ReferenceFxMatcherError(
            f"v4 contract requires {SAMPLE_RATE} Hz, got {sample_rate}"
        )
    processor = _select_official_processor("eq", sample_rate)
    if processor.num_params != EQ_PARAMETER_COUNT:
        raise ReferenceFxMatcherError(
            f"official EQ contract changed: {processor.num_params} parameters"
        )
    ranges = {
        str(name): (float(bounds[0]), float(bounds[1]))
        for name, bounds in processor.param_ranges.items()
    }
    if len(ranges) != EQ_PARAMETER_COUNT:
        raise ReferenceFxMatcherError(
            f"official EQ range contract changed: {len(ranges)} fields"
        )
    return ranges


def _validate_audio(audio: torch.Tensor, name: str) -> None:
    if not isinstance(audio, torch.Tensor):
        raise ReferenceFxMatcherError(f"{name} must be a torch.Tensor")
    if audio.ndim != 3 or audio.shape[1] not in (1, 2) or audio.shape[-1] < 1:
        raise ReferenceFxMatcherError(
            f"{name} must have shape [batch, 1|2, samples], got {tuple(audio.shape)}"
        )
    if not audio.is_floating_point():
        raise ReferenceFxMatcherError(f"{name} must use a floating dtype")
    if not torch.isfinite(audio).all():
        raise ReferenceFxMatcherError(f"{name} contains NaN or Inf")


def _inverse_sigmoid(normalized: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(normalized).all():
        raise ReferenceFxMatcherError("initial parameters contain NaN or Inf")
    if bool(((normalized <= 0.0) | (normalized >= 1.0)).any()):
        raise ReferenceFxMatcherError("initial normalized parameters must be inside (0, 1)")
    return torch.log(normalized) - torch.log1p(-normalized)


class BoundedFxParameters(nn.Module):
    def __init__(
        self,
        mode: str,
        batch_size: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        initial_normalized: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        count = parameter_count(mode)
        if batch_size < 1:
            raise ReferenceFxMatcherError("batch_size must be positive")
        self.mode = mode
        expected = (batch_size, count)
        if initial_normalized is None:
            logits = torch.zeros(expected, device=device, dtype=dtype)
        else:
            value = torch.as_tensor(initial_normalized, device=device, dtype=dtype)
            if tuple(value.shape) != expected:
                raise ReferenceFxMatcherError(
                    f"initial parameters for {mode} must have shape {expected}, got {tuple(value.shape)}"
                )
            logits = _inverse_sigmoid(value)
        self.logits = nn.Parameter(logits)

    def normalized(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


class OfficialFxRenderer(nn.Module):
    effect_order = ("eq", "reverb")

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        *,
        device: torch.device | str = "cpu",
        reverb_seed: int = DEFAULT_REVERB_SEED,
    ) -> None:
        super().__init__()
        if sample_rate != SAMPLE_RATE:
            raise ReferenceFxMatcherError(
                f"v4 contract requires {SAMPLE_RATE} Hz, got {sample_rate}"
            )
        self.sample_rate = int(sample_rate)
        self.device_name = str(device)
        self.reverb_seed = int(reverb_seed)

        self.eq = _select_official_processor("eq", self.sample_rate)
        self.reverb = _OfficialReverb(
            sample_rate=self.sample_rate,
            device=self.device_name,
        )
        if self.eq.num_params != EQ_PARAMETER_COUNT:
            raise ReferenceFxMatcherError(
                f"official EQ contract changed: {self.eq.num_params} parameters"
            )
        if self.reverb.num_params != REVERB_PARAMETER_COUNT:
            raise ReferenceFxMatcherError(
                f"official reverb contract changed: {self.reverb.num_params} parameters"
            )

    def _render_reverb(self, audio: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(self.reverb_seed)
            return self.reverb.process_normalized(audio, parameters)

    def render(
        self,
        audio: torch.Tensor,
        normalized_parameters: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        _validate_audio(audio, "audio")
        count = parameter_count(mode)
        expected = (audio.shape[0], count)
        if tuple(normalized_parameters.shape) != expected:
            raise ReferenceFxMatcherError(
                f"{mode} parameters must have shape {expected}, got {tuple(normalized_parameters.shape)}"
            )
        if not normalized_parameters.is_floating_point():
            raise ReferenceFxMatcherError("parameters must use a floating dtype")
        if not torch.isfinite(normalized_parameters).all():
            raise ReferenceFxMatcherError("parameters contain NaN or Inf")
        if bool(((normalized_parameters < 0.0) | (normalized_parameters > 1.0)).any()):
            raise ReferenceFxMatcherError("normalized parameters must stay inside [0, 1]")

        if mode == "eq":
            return self.eq.process_normalized(audio, normalized_parameters)
        if mode == "reverb":
            return self._render_reverb(audio, normalized_parameters)

        eq_parameters = normalized_parameters[:, :EQ_PARAMETER_COUNT]
        reverb_parameters = normalized_parameters[:, EQ_PARAMETER_COUNT:]
        after_eq = self.eq.process_normalized(audio, eq_parameters)
        return self._render_reverb(after_eq, reverb_parameters)

    def physical_parameters(
        self, normalized_parameters: torch.Tensor, mode: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        count = parameter_count(mode)
        if normalized_parameters.ndim != 2 or normalized_parameters.shape[1] != count:
            raise ReferenceFxMatcherError(
                f"{mode} parameters need {count} columns"
            )
        result: dict[str, dict[str, torch.Tensor]] = {}
        if mode in ("eq", "eq_reverb"):
            eq_values = normalized_parameters[:, :EQ_PARAMETER_COUNT]
            result["eq"] = self.eq.denormalize_param_dict(
                self.eq.extract_param_dict(eq_values)
            )
        if mode in ("reverb", "eq_reverb"):
            start = EQ_PARAMETER_COUNT if mode == "eq_reverb" else 0
            reverb_values = normalized_parameters[:, start : start + REVERB_PARAMETER_COUNT]
            result["reverb"] = self.reverb.denormalize_param_dict(
                self.reverb.extract_param_dict(reverb_values)
            )
        return result


def freeze_fx_encoder(encoder: nn.Module) -> nn.Module:
    encoder.eval()
    for weight in encoder.parameters():
        weight.requires_grad_(False)
        weight.grad = None
    return encoder


def _encode(encoder: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    if not hasattr(encoder, "get_fx_embedding"):
        raise ReferenceFxMatcherError("encoder needs get_fx_embedding(audio)")
    embedding = encoder.get_fx_embedding(audio)
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ReferenceFxMatcherError("encoder embedding must have shape [batch, dimensions]")
    if not torch.isfinite(embedding).all():
        raise ReferenceFxMatcherError("encoder returned NaN or Inf")
    return embedding


def reference_embedding(encoder: nn.Module, reference_audio: torch.Tensor) -> torch.Tensor:
    _validate_audio(reference_audio, "reference_audio")
    freeze_fx_encoder(encoder)
    with torch.no_grad():
        return _encode(encoder, reference_audio).detach()


def cosine_reference_loss(
    encoder: nn.Module,
    candidate_audio: torch.Tensor,
    detached_reference_embedding: torch.Tensor,
) -> torch.Tensor:
    candidate_embedding = _encode(encoder, candidate_audio)
    if candidate_embedding.shape != detached_reference_embedding.shape:
        raise ReferenceFxMatcherError(
            "candidate and reference embedding shapes differ: "
            f"{tuple(candidate_embedding.shape)} vs {tuple(detached_reference_embedding.shape)}"
        )
    return (1.0 - F.cosine_similarity(
        candidate_embedding, detached_reference_embedding, dim=-1
    )).mean()


@dataclass(frozen=True)
class MatchResult:
    mode: str
    initial_normalized_parameters: torch.Tensor
    normalized_parameters: torch.Tensor
    physical_parameters: dict[str, dict[str, torch.Tensor]]
    rendered_audio: torch.Tensor
    loss_trace: tuple[float, ...]
    steps: int = MATCH_STEPS


def run_reference_match(
    encoder: nn.Module,
    reference_audio: torch.Tensor,
    target_audio: torch.Tensor,
    mode: str,
    *,
    renderer: OfficialFxRenderer | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    initial_normalized: torch.Tensor | None = None,
) -> MatchResult:
    _validate_audio(reference_audio, "reference_audio")
    _validate_audio(target_audio, "target_audio")
    if reference_audio.shape[0] != target_audio.shape[0]:
        raise ReferenceFxMatcherError("reference and target batch sizes must match")
    if reference_audio.device != target_audio.device:
        raise ReferenceFxMatcherError("reference and target must use the same device")
    if learning_rate <= 0.0:
        raise ReferenceFxMatcherError("learning_rate must be positive")

    renderer = renderer or OfficialFxRenderer(
        sample_rate=SAMPLE_RATE, device=target_audio.device
    )
    freeze_fx_encoder(encoder)
    target_embedding = reference_embedding(encoder, reference_audio)
    parameters = BoundedFxParameters(
        mode,
        batch_size=target_audio.shape[0],
        device=target_audio.device,
        dtype=target_audio.dtype,
        initial_normalized=initial_normalized,
    )
    initial = parameters.normalized().detach().clone()
    optimizer = torch.optim.Adam(parameters.parameters(), lr=float(learning_rate))

    trace: list[float] = []

    for _ in range(MATCH_STEPS):
        optimizer.zero_grad(set_to_none=True)
        rendered = renderer.render(target_audio, parameters.normalized(), mode)
        loss = cosine_reference_loss(encoder, rendered, target_embedding)
        if not torch.isfinite(loss):
            raise ReferenceFxMatcherError("matching loss became NaN or Inf")
        loss.backward()
        if parameters.logits.grad is None or not torch.isfinite(parameters.logits.grad).all():
            raise ReferenceFxMatcherError("parameter gradient is missing, NaN, or Inf")
        optimizer.step()
        trace.append(float(loss.detach()))

    final_normalized = parameters.normalized().detach().clone()
    with torch.no_grad():
        final_audio = renderer.render(target_audio, final_normalized, mode)
        final_loss = cosine_reference_loss(encoder, final_audio, target_embedding)
    trace.append(float(final_loss))
    physical = renderer.physical_parameters(final_normalized, mode)
    return MatchResult(
        mode=mode,
        initial_normalized_parameters=initial,
        normalized_parameters=final_normalized,
        physical_parameters={
            effect: {name: value.detach().clone() for name, value in values.items()}
            for effect, values in physical.items()
        },
        rendered_audio=final_audio.detach().clone(),
        loss_trace=tuple(trace),
    )
