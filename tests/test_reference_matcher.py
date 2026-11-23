from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import reference_matcher as fx


class FakeFxEncoder(nn.Module):
    def __init__(self, dimensions: int = 8) -> None:
        super().__init__()
        self.projection = nn.Linear(8, dimensions, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dimensions, 8))

    def get_fx_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        mono = audio.mean(dim=1)
        first = mono[..., 1:] - mono[..., :-1]
        second = first[..., 1:] - first[..., :-1]
        features = torch.stack(
            (
                mono.mean(-1),
                mono.square().mean(-1),
                mono.abs().mean(-1),
                mono.amax(-1),
                mono.amin(-1),
                first.abs().mean(-1),
                first.square().mean(-1),
                second.abs().mean(-1),
            ),
            dim=-1,
        )
        return F.normalize(self.projection(features), dim=-1)


def short_stereo(samples: int = 512) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7001 + samples)
    return torch.randn(1, 2, samples, generator=generator) * 0.03


def non_neutral_parameters(count: int) -> torch.Tensor:
    return torch.linspace(0.22, 0.78, count).unsqueeze(0).requires_grad_(True)


def test_official_checkout_and_exact_parameter_contracts():
    assert fx.OFFICIAL_REPO.is_dir()
    assert (fx.OFFICIAL_REPO / "fx_chain" / "fx_processors.py").is_file()
    assert fx.PARAM_COUNTS == {"eq": 18, "reverb": 25, "eq_reverb": 43}

    renderer = fx.OfficialFxRenderer()
    assert renderer.eq.num_params == 18
    assert renderer.reverb.num_params == 25
    assert len(renderer.eq.param_ranges) == 18
    assert len(renderer.reverb.param_ranges) == 25
    assert renderer.reverb.__class__.__module__ == "fx_chain.fx_processors"

    ranges = fx.official_eq_parameter_ranges()
    assert len(ranges) == 18
    assert ranges["low_shelf_gain_db"] == (-12.0, 12.0)
    assert ranges["band2_cutoff_freq"] == (8000.0, 12000.0)
    assert ranges["high_shelf_q_factor"] == (0.5, 5.0)


@pytest.mark.parametrize("mode,count", fx.PARAM_COUNTS.items())
def test_bounded_sigmoid_parameter_shape_and_range(mode: str, count: int):
    parameters = fx.BoundedFxParameters(mode, batch_size=2)
    normalized = parameters.normalized()
    assert normalized.shape == (2, count)
    assert bool((normalized > 0.0).all())
    assert bool((normalized < 1.0).all())


def test_reference_is_detached_while_candidate_gradient_survives():
    encoder = FakeFxEncoder()
    reference = short_stereo(256)
    target = short_stereo(257)[:, :, :256].requires_grad_(True)

    reference_value = fx.reference_embedding(encoder, reference)
    loss = fx.cosine_reference_loss(encoder, target, reference_value)
    loss.backward()

    assert not reference_value.requires_grad
    assert target.grad is not None
    assert torch.isfinite(target.grad).all()
    assert float(target.grad.abs().sum()) > 0.0
    assert not encoder.training
    assert all(not weight.requires_grad for weight in encoder.parameters())
    assert all(weight.grad is None for weight in encoder.parameters())


def test_eq_all_18_gradients_are_finite_and_nonzero():
    renderer = fx.OfficialFxRenderer()
    audio = short_stereo(1024)
    parameters = non_neutral_parameters(18)
    output = renderer.render(audio, parameters, "eq")
    weights = torch.linspace(0.3, 1.3, output.numel()).reshape_as(output)
    (output * weights).mean().backward()

    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert int((parameters.grad != 0.0).sum()) == 18


def test_reverb_all_25_gradients_are_finite_nonzero_and_ir_is_fixed():
    renderer = fx.OfficialFxRenderer(reverb_seed=9817)
    audio = short_stereo(256)
    parameters = non_neutral_parameters(25)

    first = renderer.render(audio, parameters, "reverb")
    second = renderer.render(audio, parameters, "reverb")
    assert torch.equal(first, second)

    weights = torch.linspace(0.3, 1.3, first.numel()).reshape_as(first)
    (first * weights).mean().backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert int((parameters.grad != 0.0).sum()) == 25


def test_reverb_render_does_not_advance_callers_random_state():
    renderer = fx.OfficialFxRenderer(reverb_seed=123)
    audio = short_stereo(64)
    parameters = torch.full((1, 25), 0.5)
    torch.manual_seed(44)
    before = torch.random.get_rng_state().clone()
    renderer.render(audio, parameters, "reverb")
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_combined_mode_uses_eq_then_reverb_and_all_43_parameters():
    renderer = fx.OfficialFxRenderer(reverb_seed=55)
    audio = short_stereo(96)
    parameters = non_neutral_parameters(43)
    combined = renderer.render(audio, parameters, "eq_reverb")

    after_eq = renderer.render(audio, parameters[:, :18], "eq")
    separate = renderer.render(after_eq, parameters[:, 18:], "reverb")
    assert torch.equal(combined, separate)

    physical = renderer.physical_parameters(parameters, "eq_reverb")
    assert set(physical) == {"eq", "reverb"}
    assert len(physical["eq"]) == 18
    assert len(physical["reverb"]) == 25

    weights = torch.linspace(0.3, 1.3, combined.numel()).reshape_as(combined)
    (combined * weights).mean().backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert int((parameters.grad != 0.0).sum()) == 43


def test_matcher_is_fixed_at_60_steps_and_returns_final_iteration():
    encoder = FakeFxEncoder()
    renderer = fx.OfficialFxRenderer()
    reference = short_stereo(384)
    target = short_stereo(385)[:, :, :384]

    result = fx.run_reference_match(
        encoder,
        reference,
        target,
        "eq",
        renderer=renderer,
        learning_rate=0.01,
        initial_normalized=torch.full((1, 18), 0.55),
    )

    assert result.steps == 60 == fx.MATCH_STEPS
    assert len(result.loss_trace) == 61
    assert result.normalized_parameters.shape == (1, 18)
    assert result.rendered_audio.shape == target.shape
    assert set(result.physical_parameters) == {"eq"}
    assert all(not weight.requires_grad for weight in encoder.parameters())
    replayed = renderer.render(target, result.normalized_parameters, "eq")
    assert torch.equal(result.rendered_audio, replayed)
    detached_reference = fx.reference_embedding(encoder, reference)
    exact_final_loss = fx.cosine_reference_loss(
        encoder, replayed, detached_reference
    )
    assert result.loss_trace[-1] == pytest.approx(float(exact_final_loss))


def test_wrong_parameter_count_fails_instead_of_clamping():
    renderer = fx.OfficialFxRenderer()
    with pytest.raises(fx.ReferenceFxMatcherError, match="shape"):
        renderer.render(short_stereo(64), torch.full((1, 17), 0.5), "eq")
    with pytest.raises(fx.ReferenceFxMatcherError, match="inside"):
        renderer.render(short_stereo(64), torch.full((1, 18), 1.2), "eq")


def test_only_new_module_paths_are_under_workspace():
    root = Path(__file__).resolve().parents[1] / "src"
    assert Path(fx.__file__).resolve() == root / "reference_matcher.py"
