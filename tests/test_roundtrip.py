from __future__ import annotations

import pytest

import roundtrip_common as roundtrip


def test_model_identity_and_separation_are_preserved():
    assert roundtrip.MODEL == roundtrip.PLANNER_MODEL
    assert roundtrip.MODEL != roundtrip.AUTHOR_MODEL


@pytest.mark.parametrize(
    "value",
    [
        "Move from the original to the treatment after 20%.",
        "Move from the original to the treatment after two passages.",
        "Move from the original to the treatment after half the passage.",
        "Move from the original to the treatment after several bars.",
        "At 01:20 move from the original to the treatment.",
    ],
)
def test_validator_rejects_only_explicit_numeric_timing(value):
    with pytest.raises(roundtrip.InstructionValidationError):
        roundtrip.validate_natural_instruction(value)


def test_validator_requires_original_and_treatment_concepts():
    with pytest.raises(roundtrip.InstructionValidationError, match="concepts"):
        roundtrip.validate_natural_instruction("Let the sound gradually change near the close.")


def test_prompt_adds_only_generic_semantic_contract():
    assert roundtrip.PLANNER_SYSTEM_PROMPT.startswith(roundtrip.BASE_PLANNER_SYSTEM_PROMPT)
    appended = roundtrip.PLANNER_SYSTEM_PROMPT[len(roundtrip.BASE_PLANNER_SYSTEM_PROMPT):]
    for text in (
        "distinct basis endpoints",
        "do not name the current state",
        "may be reused",
        "already defines the opening state",
        "target_amount of 0 selects the source basis",
        "normalized by the sum across all events",
        "first control coordinate measured",
        "remaining distance from handle1",
        "exact, ordered, non-overlapping",
    ):
        assert text in appended
    assert "curve_" not in appended
    assert "registered treatment" not in appended
