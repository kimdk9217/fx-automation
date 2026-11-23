# Music Structure-Aware FX Automation from Natural Language

Turns a natural-language instruction into time-varying EQ and reverb automation for one section of a song.

Example instruction: *"start warm, then slowly move toward the reference sound and hold it."*
The system finds the section and its bars, works out what "warm" and the reference sound mean as effect
settings, plans when the sound should change and when it should hold, and renders the result as an
editable automation curve and processed audio.

## How it works

1. **Structure analysis.** [All-In-One](https://github.com/mir-aidj/all-in-one) gives beats, downbeats and
   labelled sections (`src/automation_fx/structure.py`).
2. **Endpoints.** Each target sound becomes an *endpoint*, a full set of EQ/reverb parameters.
   - Text: an LLM maps the description to parameter values (`src/text_endpoints.py`).
   - Reference audio: parameters are fitted so the processed target matches the reference in a frozen
     [Fx-Encoder++](https://github.com/SonyResearch/Fx-Encoder_PlusPlus) embedding space (`src/reference_matcher.py`).
3. **Planning.** An LLM turns the instruction into an ordered list of events: `set_state` (move toward
   an endpoint) and `hold_current` (stay). It sees the section, its bars and a DSP summary of each
   endpoint, but no audio (`src/planner.py`, `src/dsp_summary.py`).
4. **Curve compilation.** Each event is a cubic Bézier segment with monotonic handles; the segments
   share boundary values, so the curve stays continuous and editable (`src/automation_fx/bezier.py`).
5. **Rendering.** Endpoint weights are applied sample by sample: a six-band EQ through
   [TorchLPC](https://github.com/DiffAPF/torchlpc) and a reverb send (`src/renderer.py`,
   `src/automation_fx/eq.py`, `src/automation_fx/reverb.py`).

Every run writes its inputs, requests, raw responses and outputs to disk with hashes, and a separate
step verifies the finished run against them.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Pipeline modules and command-line entry points |
| `src/automation_fx/` | DSP building blocks: EQ, reverb, Bézier curves, structure analysis |
| `src/eval1_*.py` | Evaluation 1: direction of style transitions |
| `src/eval2_*.py`, `src/roundtrip_common.py` | Evaluation 2: reconstructing curves from text |
| `examples/` | Offline examples that need no API key and no dataset |
| `tests/` | Unit tests and an offline end-to-end test |
| `results/` | Recorded outputs of both evaluations |
| `vendor/Fx-Encoder_PlusPlus` | Fx-Encoder++ source, pinned as a git submodule |

## Setup

Python 3.10 or newer.

```sh
git clone --recursive <this repository>
cd <repository folder>
python -m pip install -r requirements.txt
```

If you cloned without `--recursive`, run `git submodule update --init`.
The submodule is pinned to commit `7b1e1f7efaa8a7c212ae4d62c01a5bac5f61c47b`; the reference matcher checks this.

## Quick check (no API key, no dataset)

```sh
python -m pytest -q
python examples/curve_demo.py --output output/curve_demo
python examples/offline_pipeline.py --output output/offline_pipeline
python examples/verify_recorded_results.py
```

- `curve_demo.py` compiles a fixed plan into a curve and applies the EQ to a synthetic signal.
- `offline_pipeline.py` runs the whole pipeline (prepare, freeze the semantic case, execute, verify,
  evaluate) with a fake LLM and a fake encoder. Everything else is the real code.
- `verify_recorded_results.py` recomputes the reported numbers from `results/`.

## Running on real audio

You need:

- an API key for the LLM, in `AUTOMATION_LLM_API_KEY`
- the Fx-Encoder++ checkpoint, in the Hugging Face cache under `FXENCODER_CACHE` (default `.cache/fxencoder_plusplus`)
- `pip install allin1` for structure analysis (it also needs NATTEN)

Steps:

```sh
# 1. Structure of the full mix
python src/automation_fx/structure.py song.wav runs/structure.json

# 2. Write a manifest (format below), then prepare the run
python src/prepare_run.py prepare --manifest runs/manifest.json --output runs/prepared

# 3. Freeze the expected outcome of the instruction: semantic_case.prepare_case(...)
#    (see examples/offline_pipeline.py for a full call)

# 4. Bind the prepared run and the semantic case, then execute
python src/execute_run.py prepare-execution --prepared-run runs/prepared --semantic-case runs/semantic_case --output runs/execution
python src/execute_run.py execute --prepared runs/execution --output runs/run --reviewed
python src/execute_run.py verify-completed --run runs/run

# 5. Score the run against the semantic case: evaluate_run.write_evaluation(...)
```

`execute` makes one LLM call per text endpoint and one call for the automation plan. Runs never retry
or repair a model response; a failed call is recorded as a failure.

Manifest format:

```json
{
  "schema_version": "dynamic-user-endpoint-manifest-v3",
  "target": {
    "raw_stem_path": "stems/vocals.wav",
    "structure_path": "runs/structure.json",
    "stem": "vocals",
    "section": "chorus",
    "occurrence": 1
  },
  "automation_instruction": "start warm, then slowly move toward the reference sound and hold it",
  "renderer_mode": "eq",
  "endpoints": [
    {"kind": "natural_language", "text": "a warm, dark vocal", "effect_mode": "eq"},
    {"kind": "processed_reference", "audio_path": "reference.wav", "effect_mode": "eq"}
  ]
}
```

`stem` is one of `vocals`, `drums`, `bass`, `other`; audio must be 44.1 kHz. `renderer_mode` and
`effect_mode` are `eq`, `reverb` or `eq_reverb`. Relative paths are resolved from the manifest's folder.

## Evaluations

**Evaluation 1, style transitions.** Ten transitions between SocialFX style words (for example warm → bright)
on dry piano passages from MedleyDB. The score is whether the rendered audio moves from the source
style toward the target style. Result: 9 of 10 pairs move in the right direction, median Spearman
ρ = 0.525. This measures direction only, not perceptual quality.

**Evaluation 2, curve reconstruction.** Thirty synthetic curves, each described in three ways by a
separate LLM; the planner rebuilds the curve from the text alone. Result: the sequence of rises,
falls and holds is exactly right in 71 of 90 cases (78.9%); mean curve error 0.152, boundary error 0.171.

The evaluation scripts (`eval1_prepare/plan/score/summarize`, `eval2_roundtrip`, `eval2_baselines`)
expect the original intermediate run folders and the licensed audio, which are not included.
The recorded outputs are in `results/`.

## Environment variables

| Variable | Default | Used for |
| --- | --- | --- |
| `AUTOMATION_LLM_API_KEY` | none | LLM calls |
| `FXENCODER_OFFICIAL_REPO` | `vendor/Fx-Encoder_PlusPlus` | Fx-Encoder++ source |
| `FXENCODER_CACHE` | `.cache/fxencoder_plusplus` | Fx-Encoder++ checkpoint |
| `MEDLEYDB_ARCHIVE` | `data/MedleyDB_V1.tar.gz` | Evaluation 1 audio |

## Third-party code and data

Fx-Encoder++, All-In-One, TorchLPC, DASP-PyTorch, SocialFX and MedleyDB keep their own licenses.
Fx-Encoder++ is linked as a submodule (CC BY-NC 4.0), not copied into this repository. No audio or
model weights are included.
