"""Evaluation 1 settings: models, style words and the ten transition pairs."""

from __future__ import annotations

MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
RUNS_PER_INSTRUCTION = 3

WORDS = ("warm", "soft", "harsh", "calm", "loud", "bright", "heavy")

EXPECTED_SOCIALFX_COUNTS = {
    "warm": 74, "bright": 22, "soft": 69, "loud": 26,
    "harsh": 32, "calm": 29, "heavy": 21,
}


TRANSITION_PAIRS = (
    ("warm", "heavy"), ("loud", "heavy"), ("bright", "warm"), ("heavy", "calm"),
    ("soft", "loud"), ("calm", "loud"), ("harsh", "calm"), ("harsh", "soft"),
    ("warm", "bright"), ("heavy", "harsh"),
)


MEASURED_PAIR_SEPARABILITY = {
    ("warm", "heavy"): 0.439, ("loud", "heavy"): 0.918, ("bright", "warm"): 1.361,
    ("heavy", "calm"): 1.274, ("soft", "loud"): 1.159, ("calm", "loud"): 1.296,
    ("harsh", "calm"): 0.774, ("harsh", "soft"): 0.794, ("warm", "bright"): 1.361,
    ("heavy", "harsh"): 1.437,
}


RUN_VERSION = "transition10-dry-acoustic-piano-v1"
N_WINDOWS = 16

PRIMARY_ALPHA = 0.10
CONVENTIONAL_ALPHA = 0.05
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260826
WINDOW_SECONDS = 16.0
WINDOW_SEARCH_HOP_SECONDS = 1.0


TRANSITION_TEMPLATE = (
    "Across the entire selected piano passage, start the piano sounding {a} "
    "and gradually change it so that by the end it sounds {b}."
)


def _key(a: str, b: str) -> str:
    return f"T_{a}_to_{b}"


INSTRUCTIONS = {
    _key(a, b): TRANSITION_TEMPLATE.format(a=a, b=b)
    for a, b in TRANSITION_PAIRS
}
INSTRUCTION_KEYS = tuple(INSTRUCTIONS)
EXPECTED = {
    _key(a, b): {
        "kind": "transition",
        "source_word": a,
        "target_word": b,
        "separability": MEASURED_PAIR_SEPARABILITY[(a, b)],
        "scored": True,
    }
    for a, b in TRANSITION_PAIRS
}


PIANO_CASES = (
    ("ClaraBerryAndWooldog_AirTraffic", "S06"),
    ("ClaraBerryAndWooldog_WaltzForMyVictims", "S03"),
    ("Lushlife_ToynbeeSuite", "S23"),
    ("MatthewEntwistle_AnEveningWithOliver", "S11"),
    ("MatthewEntwistle_DontYouEver", "S02"),
    ("MatthewEntwistle_TheArch", "S18"),
    ("MusicDelta_BebopJazz", "S03"),
    ("MusicDelta_CoolJazz", "S03"),
    ("MusicDelta_ModalJazz", "S03"),
    ("MusicDelta_SwingJazz", "S03"),
    ("SweetLights_YouLetMeDown", "S09"),
)

WINDOW_RULE = (
    "Convert the single RAW piano file to stereo. Among 16.0-second windows "
    "whose starts lie on one-second boundaries, choose the window with the "
    "largest mean-square amplitude; resolve ties by the earliest start. The "
    "selection uses input amplitude only, before any LLM call, rendering, "
    "SocialFX feature, or score. Apply a common safety gain only if peak exceeds 0.95."
)

KNOWN_LIMITATIONS = (
    "MedleyDB RAW is the least-processed delivered source, not proof that no upstream recording coloration exists.",
    "The 11 cases are acoustic-piano stems only and do not establish results for electric piano, tack piano, or other instruments.",
    "The selected passages are fixed by input activity, not musical structure. This run evaluates style trajectory on piano, not structure localization.",
    "The same 30 piano plans are rendered on all 11 inputs, so tracks and windows are not treated as independent LLM samples.",
)
