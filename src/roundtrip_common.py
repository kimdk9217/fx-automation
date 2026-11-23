"""Shared pieces of the curve reconstruction evaluation: prompts, validation and scoring."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import llm
import planner
import prepare_run as production_prepare

AUTHOR_MODEL = "gpt-5.6-sol"
PLANNER_MODEL = "gpt-5.6-terra"

BOOTSTRAP_SEED = 20260827
N_BOOTSTRAP = 10_000
N_SAMPLES = 1001
N_CURVES = 30

ORACLE_DISTANCE_TIE_TOLERANCE = 1e-12
ORIGINAL_ID = "endpoint_000"
TREATMENT_ID = "endpoint_001"
TARGET_STEM = "vocals"
TARGET_SECTION = "evaluation_section"

BASE_PLANNER_SYSTEM_PROMPT = production_prepare.AUTOMATION_SYSTEM_PROMPT

if AUTHOR_MODEL == PLANNER_MODEL:
    raise RuntimeError("instruction author and automation planner models must differ")

FAMILY_EVENT_COUNTS: dict[str, tuple[int, ...]] = {
    "progressive_rise": (1, 2, 3, 4, 5),
    "progressive_fall": (3, 4, 5, 6, 7),
    "arched_return": (4, 5, 6, 7, 8),
    "valley_return": (2, 3, 4, 5, 6),
    "alternating_partial": (1, 3, 5, 7, 8),
    "complex_plateaus": (4, 5, 6, 7, 8),
}


class RoundTripError(RuntimeError):
    pass


class _Registry:
    mode = "eq"
    endpoint_ids = (ORIGINAL_ID, TREATMENT_ID)


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planner_schema() -> dict[str, Any]:
    return planner.make_plan_schema(
        registry=_Registry(),
        allowed_stems=[TARGET_STEM],
        allowed_sections=[TARGET_SECTION],
        allowed_section_occurrences=[(TARGET_SECTION, 1)],
        locked_stem=TARGET_STEM,
        locked_section=TARGET_SECTION,
        locked_occurrence=1,
    )


def _native_request(payload: Mapping[str, Any], reasoning_effort: str) -> dict[str, Any]:
    response_format = deepcopy(payload["response_format"])
    response_format["json_schema"]["schema"] = llm.to_openai_strict(
        response_format["json_schema"]["schema"]
    )
    body: dict[str, Any] = {
        "model": payload["model"],
        "messages": deepcopy(payload["messages"]),
        "response_format": response_format,
        "reasoning_effort": reasoning_effort,
    }
    return {
        "endpoint": f"{llm.DEFAULT_BASE_URL}/chat/completions",
        "body": body,
        "api_key_recorded": False,
    }


def _amount_from_event(event: Mapping[str, Any]) -> float | None:
    if event["relation"] == "hold_current":
        return None
    source = str(event["source_endpoint_id"])
    target = str(event["target_endpoint_id"])
    amount = float(event["target_amount"])
    if source == ORIGINAL_ID and target == TREATMENT_ID:
        return amount
    if source == TREATMENT_ID and target == ORIGINAL_ID:
        return 1.0 - amount
    raise RoundTripError("plan contains unsupported endpoint pair")


def topology_from_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    current = 0.0 if plan["initial_endpoint_id"] == ORIGINAL_ID else 1.0
    result = []
    for event in plan["events"]:
        target = _amount_from_event(event)
        if target is None:
            result.append("hold")
            continue
        difference = target - current
        if difference > 1e-9:
            result.append("up")
        elif difference < -1e-9:
            result.append("down")
        else:
            raise RoundTripError("set_state event does not change state")
        current = target
    return tuple(result)


def _resolved(plan: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    weights, report = planner.evaluate_endpoint_schedule(
        plan, registry=_Registry(), n_samples=N_SAMPLES
    )
    index = list(_Registry.endpoint_ids).index(TREATMENT_ID)
    return weights[index].astype(np.float64), report


def _change_points(report: Mapping[str, Any]) -> list[float]:
    events = list(report["events"])
    return [float(event["normalized_stop_time"]) for event in events[:-1]]


def _planner_context(instruction: str) -> dict[str, Any]:
    return {
        "INSTRUCTION": instruction,
        "LOCKED_TARGET": {
            "stem": TARGET_STEM,
            "section": TARGET_SECTION,
            "occurrence": 1,
        },
        "LOCKED_EFFECT_MODE": _Registry.mode,
        "ENDPOINTS": [
            {
                "endpoint_id": ORIGINAL_ID,
                "role": "dry",
                "caller_description_exact": "original sound",
            },
            {
                "endpoint_id": TREATMENT_ID,
                "role": "caller_natural_language",
                "caller_description_exact": "registered treatment",
            },
        ],
        "VISIBILITY_CONTRACT": {
            "instruction": True,
            "endpoint_names_and_roles": True,
            "ground_truth_curve": False,
            "numeric_timeline": False,
            "audio": False,
            "reference_audio": False,
            "embeddings": False,
            "dsp_parameters": False,
        },
    }


def _raw_content(response: Mapping[str, Any]) -> Any:
    if response.get("native_meta", {}).get("finish_reason") != "stop":
        raise RoundTripError("structured response did not finish with stop")
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)


def _assert_response_model(response: Mapping[str, Any], requested_model: str) -> None:
    actual = response.get("native_meta", {}).get("model")
    if not isinstance(actual, str) or not (
        actual == requested_model or actual.startswith(f"{requested_model}-")
    ):
        raise RoundTripError(
            f"response model mismatch: requested {requested_model!r}, received {actual!r}"
        )


def _penalized_change_point_error(expected: list[float], predicted: list[float]) -> float:
    paired = min(len(expected), len(predicted))
    total = sum(abs(float(expected[index]) - float(predicted[index])) for index in range(paired))
    total += abs(len(expected) - len(predicted)) * 1.0
    return float(total / max(len(expected), len(predicted), 1))


def invalid_score(case: Mapping[str, Any], reason: Any) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "curve_id": case["curve_id"],
        "family": case["family"],
        "paraphrase_index": case["paraphrase_index"],
        "valid_plan": False,
        "failure": reason,
        "metrics": {
            "event_sequence_exact": False,
            "curve_mae": 1.0,
            "penalized_change_point_error": 1.0,
            "initial_state_exact": False,
            "final_state_absolute_error": 1.0,
            "final_state_within_0_01": False,
            "correct_oracle_rank": N_CURVES,
            "correct_oracle_top1": False,
            "correct_oracle_tie_count": 0,
        },
        "predicted_curve_1001": None,
        "predicted_topology": None,
        "predicted_change_points": None,
    }


def score_prediction(
    case: Mapping[str, Any], plan: Mapping[str, Any], ground_truth: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {row["curve_id"]: row for row in ground_truth}
    oracle = by_id[case["curve_id"]]
    predicted_curve, report = _resolved(plan)
    predicted_topology = list(topology_from_plan(plan))
    predicted_change_points = _change_points(report)
    expected_curve = np.asarray(oracle["curve_1001"], dtype=np.float64)
    curve_mae = float(np.mean(np.abs(predicted_curve - expected_curve)))
    distances = {
        row["curve_id"]: float(
            np.mean(
                np.abs(predicted_curve - np.asarray(row["curve_1001"], dtype=np.float64))
            )
        )
        for row in ground_truth
    }
    correct_distance = distances[case["curve_id"]]
    minimum_distance = min(distances.values())
    rank = 1 + sum(
        distance < correct_distance - ORACLE_DISTANCE_TIE_TOLERANCE
        for distance in distances.values()
    )
    tie_count = sum(
        abs(distance - correct_distance) <= ORACLE_DISTANCE_TIE_TOLERANCE
        for distance in distances.values()
    )
    predicted_initial = float(predicted_curve[0])
    expected_initial = float(expected_curve[0])
    predicted_final = float(predicted_curve[-1])
    expected_final = float(expected_curve[-1])
    final_error = abs(predicted_final - expected_final)
    return {
        "case_id": case["case_id"],
        "curve_id": case["curve_id"],
        "family": case["family"],
        "paraphrase_index": case["paraphrase_index"],
        "valid_plan": True,
        "failure": None,
        "metrics": {
            "event_sequence_exact": predicted_topology == oracle["topology"],
            "curve_mae": curve_mae,
            "penalized_change_point_error": _penalized_change_point_error(
                list(oracle["change_points"]), predicted_change_points
            ),
            "initial_state_exact": abs(predicted_initial - expected_initial) <= 1e-12,
            "final_state_absolute_error": final_error,
            "final_state_within_0_01": final_error <= 0.01,
            "correct_oracle_rank": rank,
            "correct_oracle_top1": (
                correct_distance <= minimum_distance + ORACLE_DISTANCE_TIE_TOLERANCE
            ),
            "correct_oracle_tie_count": tie_count,
        },
        "oracle_distances_mae": distances,
        "predicted_curve_1001": predicted_curve.tolist(),
        "predicted_topology": predicted_topology,
        "predicted_change_points": predicted_change_points,
    }

_AGGREGATE_METRICS = (
    "event_sequence_exact_rate",
    "curve_mae",
    "penalized_change_point_error",
    "initial_state_accuracy",
    "final_state_absolute_error",
    "final_state_within_0_01_rate",
    "correct_oracle_top1_rate",
    "correct_oracle_mean_rank",
    "paraphrase_mae",
)


def stratified_bootstrap(curve_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    families = sorted(FAMILY_EVENT_COUNTS)
    groups = {
        family: [row for row in curve_rows if row["family"] == family]
        for family in families
    }
    if any(len(rows) != 5 for rows in groups.values()):
        raise RoundTripError("stratified bootstrap requires five curves per family")
    samples = {metric: np.empty(N_BOOTSTRAP, dtype=np.float64) for metric in _AGGREGATE_METRICS}
    for replicate in range(N_BOOTSTRAP):
        draw = []
        for family in families:
            rows = groups[family]
            draw.extend(rows[int(index)] for index in rng.integers(0, len(rows), size=len(rows)))
        for metric in _AGGREGATE_METRICS:
            samples[metric][replicate] = np.mean([float(row[metric]) for row in draw])
    return {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in curve_rows])),
            "bootstrap_95_lower": float(np.quantile(values, 0.025)),
            "bootstrap_95_upper": float(np.quantile(values, 0.975)),
        }
        for metric, values in samples.items()
    }


MODEL = "gpt-5.6-terra"

PLANNER_SYSTEM_PROMPT = BASE_PLANNER_SYSTEM_PROMPT + """

Generic blend-basis contract for this corrected follow-up:
- source_endpoint_id and target_endpoint_id are distinct basis endpoints that
  define the desired blend. They do not name the current state.
- A target_amount of 0 selects the source basis and a target_amount of 1
  selects the target basis.
- The same source/target basis pair may be reused across multiple set_state
  events with different target_amount values.
- initial_endpoint_id already defines the opening state. Do not add a redundant
  set_state event that recreates that same opening state.
- duration_weight gives an event's relative length; each event length is its
  duration_weight normalized by the sum across all events.
- Each handle1 time/value fraction is the first control coordinate measured
  from the event start/current state. Each handle2 time/value fraction is
  measured over the remaining distance from handle1 to the event end/target.
- Every instruction_evidence value must be an exact, ordered, non-overlapping
  substring of the caller instruction.
"""


class InstructionValidationError(RuntimeError):
    pass


if AUTHOR_MODEL == MODEL or MODEL != PLANNER_MODEL:
    raise RuntimeError(
        "v2 must preserve distinct v1 author/planner models and the v1 planner model"
    )


_CARDINAL_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion",
}
_FRACTION_WORDS = {
    "half", "halves", "quarter", "quarters", "third", "thirds", "fourth",
    "fourths", "fifth", "fifths", "sixth", "sixths", "seventh", "sevenths",
    "eighth", "eighths", "ninth", "ninths", "tenth", "tenths", "fraction",
    "fractions", "percent", "percentage", "percentages",
}
_COUNT_QUANTIFIERS = _CARDINAL_WORDS | {
    "a", "an", "couple", "few", "several", "many", "dozen", "dozens",
}


def validate_natural_instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InstructionValidationError("instruction must be non-empty text")
    clean = value.strip()
    normalized = unicodedata.normalize("NFKC", clean)
    lowered = normalized.casefold()
    words = re.findall(r"[a-z]+", lowered)
    word_set = set(words)
    if any(unicodedata.category(character) == "Nd" for character in normalized):
        raise InstructionValidationError("instruction contains a digit")
    if re.search(r"[%％¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]", normalized):
        raise InstructionValidationError("instruction contains percentage or numeric fraction")
    if re.search(r"\b\d{1,2}\s*:\s*\d{1,2}\b", normalized):
        raise InstructionValidationError("instruction contains a timecode")
    forbidden_cardinals = sorted(word_set & _CARDINAL_WORDS)
    if forbidden_cardinals:
        raise InstructionValidationError(
            f"instruction contains cardinal number words: {forbidden_cardinals}"
        )
    forbidden_fractions = sorted(word_set & _FRACTION_WORDS)
    if forbidden_fractions:
        raise InstructionValidationError(
            f"instruction contains fraction words: {forbidden_fractions}"
        )
    for index, word in enumerate(words):
        if word in {"bar", "bars", "beat", "beats"} and index > 0:
            if words[index - 1] in _COUNT_QUANTIFIERS:
                raise InstructionValidationError("instruction contains explicit bar or beat count")
    if "original" not in word_set or "treatment" not in word_set:
        raise InstructionValidationError(
            "instruction must contain original and treatment concepts"
        )
    return clean


def _planner_payload(instruction: str) -> dict[str, Any]:
    context = _planner_context(instruction)
    return {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_curve_roundtrip_plan_v2_corrected",
                "strict": True,
                "schema": _planner_schema(),
            },
        },
    }
