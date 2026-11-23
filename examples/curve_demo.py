"""Offline example: compile a fixed plan into a curve and apply the EQ to a synthetic signal."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dsp_summary as summaries
import planner
from automation_fx.eq import (
    BAND_ORDER,
    BandParameters,
    SixBandParameters,
    band_ranges,
    render_six_band,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/curve_demo")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing demo: {args.output}")
    sr, seconds = 44100, 4
    t = np.arange(sr * seconds) / sr
    mono = sum(0.025 * np.sin(2*np.pi*f*t) for f in (220, 880, 3520, 7040))
    mono *= np.minimum(t/0.02, 1) * np.minimum((seconds-t)/0.02, 1)
    audio = np.stack([mono, mono], axis=1).astype(np.float32)
    summary = summaries.summarize_audio(audio, sr)
    params = torch.full((1, 18), 0.5)
    proof = {
        "verification_status": "verified",
        "verification_method": "manually specified deterministic example, not an LLM output",
        "source_artifacts_verified": True,
        "normalized_parameters_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
        "dsp_summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
    }
    registry = planner.EndpointRegistry("eq", [
        planner.EndpointSpec("dry", "original", "dry", None, summary, {"source": "synthetic signal"}),
        planner.EndpointSpec("bright", "bright", "natural_language_style", params, summary, proof),
    ])
    events = []
    for i, (kind, source, target, evidence) in enumerate([
        ("set_state", "dry", "bright", "brighten"),
        ("hold_current", None, None, "hold"),
        ("set_state", "bright", "dry", "return"),
        ("hold_current", None, None, "remain"),
    ]):
        events.append({"id": f"e{i+1}", "instruction_evidence": evidence, "relation": kind,
                       "source_endpoint_id": source, "target_endpoint_id": target,
                       "target_amount": 1.0 if target else None, "duration_weight": 1.0,
                       "handle1_time_fraction": 0.3, "handle2_time_fraction": 0.7,
                       "handle1_value_fraction": 0.2, "handle2_value_fraction": 0.8})
    plan = {"schema_version": planner.SCHEMA_VERSION, "target_stem": "vocals", "effect_mode": "eq",
            "section": "chorus", "occurrence": 1, "initial_endpoint_id": "dry", "events": events}
    plan = planner.validate_plan(plan, schema=planner.make_plan_schema(registry=registry,
        allowed_stems=["vocals"], allowed_sections=["chorus"]), registry=registry,
        instruction="brighten, hold, return, remain")
    weights, report = planner.evaluate_endpoint_schedule(plan, registry=registry, n_samples=len(t))
    bright = weights[registry.endpoint_ids.index("bright")]
    ranges = band_ranges(sr)
    bands = tuple(BandParameters(b, 0, 6 if b == "high_shelf" else 0,
        np.sqrt(ranges[b]["frequency_hz"][0]*ranges[b]["frequency_hz"][1]), 0.707) for b in BAND_ORDER)
    with torch.no_grad():
        wet, _ = render_six_band(torch.from_numpy(audio), SixBandParameters(bands, sr),
                                {b: bright for b in BAND_ORDER})
    assert torch.isfinite(wet).all() and wet.abs().max() < 1
    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / "input_synthetic.wav", audio, sr, subtype="PCM_16")
    sf.write(args.output / "output_eq.wav", wet.numpy(), sr, subtype="PCM_16")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (args.output / "weights.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["time_seconds", "dry", "bright"])
        writer.writerows(zip(t[::441], weights[0, ::441], weights[1, ::441]))
    fig, ax = plt.subplots(figsize=(7, 2.6), constrained_layout=True)
    ax.plot(t[::44], bright[::44], color="#225f80")
    ax.set(xlabel="Time (s)", ylabel="Bright endpoint weight", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.2)
    fig.savefig(args.output / "curve.png", dpi=180); plt.close(fig)
    print(f"Created deterministic compiler/EQ example: {args.output}")


if __name__ == "__main__":
    main()
