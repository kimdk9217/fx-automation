"""Freezes the expected semantic outcome of one instruction before the run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import tempfile
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import endpoint_manifest as manifest
import prepare_run as prepare

EVALUATOR_VERSION = "dynamic-multi-endpoint-semantic-prereg-v3"
STATUS = "semantic_case_pre_registered_not_executed"
EVIDENCE_CLASSES = (
    "overlap_or_direct_cue_smoke",
    "lexically_disjoint_semantic_paraphrase",
)
CURVE_TRENDS = ("monotonic_toward_target",)
COMPLETION_CONSTRAINTS = ("unspecified", "full_target", "partial_target")
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_INTERNAL_ID = re.compile(r"endpoint_[0-9]{3}", re.I)
_EN_SMALL_CARDINAL = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
)
_EN_SMALL_ORDINAL = (
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth"
)
_EN_TENS_CARDINAL = r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_EN_TENS_ORDINAL = (
    r"twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|"
    r"eightieth|ninetieth"
)
_EN_POSITION = (
    rf"(?:{_EN_SMALL_CARDINAL}|{_EN_SMALL_ORDINAL}|{_EN_TENS_CARDINAL}|"
    rf"{_EN_TENS_ORDINAL}|(?:{_EN_TENS_CARDINAL})[-\s]+(?:"
    rf"{_EN_SMALL_CARDINAL}|{_EN_SMALL_ORDINAL})|last|final|"
    r"[0-9]+(?:st|nd|rd|th)?)"
)
_KO_POSITION = (
    r"(?:첫(?:\s*번째)?|[가-힣]+\s*(?:번(?:째)?|번째|째)|"
    r"[0-9]+\s*번(?:째)?|마지막|최종)"
)
_DIRECT_ORDINAL_CUE = re.compile(
    rf"(?:\b(?:the\s+)?(?:number\s+)?{_EN_POSITION}\b|"
    rf"{_KO_POSITION}|제\s*[0-9]+)",
    re.I,
)
_REFERENCE_DEICTIC = re.compile(
    r"\b(?:the\s+)?(?:processed\s+)?ref(?:erence)?\b|레퍼런스|참조\s*(?:음원|오디오)",
    re.I,
)
_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class SemanticCaseError(ValueError):
    pass


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise SemanticCaseError(f"missing frozen file: {path}")
    return {"path": str(path), "sha256": manifest.file_sha256(path)}


def _text(value: Any, where: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticCaseError(f"{where} must be non-empty text")
    if len(value) > maximum or "\x00" in value:
        raise SemanticCaseError(
            f"{where} is too long or contains NUL"
        )
    return value


def _exact(value: Any, fields: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticCaseError(
            f"{where} fields must be exactly {sorted(fields)}"
        )
    return dict(value)


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _lexical_tokens(value: str) -> list[str]:
    return sorted(
        {
            token
            for token in _LEXICAL_TOKEN.findall(_canonical_text(value))
            if len(token) >= 2
        }
    )


def _ordinal(value: Any, where: str, endpoint_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticCaseError(f"{where} must be an integer")
    result = int(value)
    if not 0 <= result <= endpoint_count:
        raise SemanticCaseError(
            f"{where} must be in 0..{endpoint_count}; zero is dry"
        )
    return result


def _evidence(value: Any, instruction: str, where: str) -> str:
    result = _text(value, where, maximum=1024)
    if result not in instruction:
        raise SemanticCaseError(
            f"{where} must be copied exactly from the caller instruction"
        )
    return result


def _endpoints(canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = canonical.get("endpoints")
    if not isinstance(rows, list) or not rows:
        raise SemanticCaseError("canonical manifest has no endpoints")
    result: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise SemanticCaseError("canonical endpoint is not an object")
        kind = raw.get("kind")
        mode = raw.get("effect_mode")
        if kind not in manifest.ENDPOINT_KINDS or mode not in manifest.EFFECT_MODES:
            raise SemanticCaseError(
                "canonical endpoint kind or effect mode changed"
            )
        caller_text = raw.get("user_description") if kind == "natural_language" else None
        if kind == "natural_language":
            caller_text = _text(caller_text, f"endpoint {ordinal} caller text")
        elif caller_text is not None:
            raise SemanticCaseError(
                "processed reference acquired an evaluator-made semantic label"
            )
        source = raw.get("source_audio") if kind == "processed_reference" else None
        if kind == "processed_reference" and not isinstance(source, Mapping):
            raise SemanticCaseError(
                "processed-reference source binding disappeared"
            )
        result.append(
            {
                "ordinal": ordinal,
                "kind": kind,
                "effect_mode": mode,
                "caller_text_exact": caller_text,
                "caller_text_utf8_sha256": (
                    None if caller_text is None else _sha_text(caller_text)
                ),
                "processed_reference_audio_sha256": (
                    source["sha256"] if isinstance(source, Mapping) else None
                ),
            }
        )
    return result


def _route(
    value: Any, *, instruction: str, endpoint_count: int
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "initial_ordinal",
            "initial_evidence_exact",
            "transition",
            "exact_event_count",
            "event_count_evidence_exact",
            "final_ordinal",
            "final_evidence_exact",
        },
        "expected_route",
    )
    transition = _exact(
        row["transition"],
        {"source_ordinal", "target_ordinal", "instruction_evidence_exact"},
        "expected_route.transition",
    )
    source = _ordinal(transition["source_ordinal"], "transition source", endpoint_count)
    target = _ordinal(transition["target_ordinal"], "transition target", endpoint_count)
    if source == target:
        raise SemanticCaseError("transition source and target must differ")

    initial = row["initial_ordinal"]
    if initial is None:
        if row["initial_evidence_exact"] is not None:
            raise SemanticCaseError(
                "unconstrained initial endpoint cannot carry initial evidence"
            )
        initial_evidence = None
    else:
        initial = _ordinal(initial, "initial ordinal", endpoint_count)
        if initial != source:
            raise SemanticCaseError(
                "single-transition scope requires constrained initial to equal source"
            )
        initial_evidence = _evidence(
            row["initial_evidence_exact"], instruction, "initial evidence"
        )

    exact_count = row["exact_event_count"]
    if exact_count is None:
        if row["event_count_evidence_exact"] is not None:
            raise SemanticCaseError(
                "unconstrained event count cannot carry count evidence"
            )
        count_evidence = None
    else:
        if (
            isinstance(exact_count, bool)
            or not isinstance(exact_count, int)
            or exact_count != 1
        ):
            raise SemanticCaseError(
                "preliminary exact event count, when instruction-entailed, must be one"
            )
        count_evidence = _evidence(
            row["event_count_evidence_exact"], instruction, "event-count evidence"
        )

    final = row["final_ordinal"]
    if final is None:
        if row["final_evidence_exact"] is not None:
            raise SemanticCaseError(
                "unconstrained final endpoint cannot carry final evidence"
            )
        final_evidence = None
    else:
        final = _ordinal(final, "final ordinal", endpoint_count)
        final_evidence = _evidence(
            row["final_evidence_exact"], instruction, "final evidence"
        )
    return {
        "scope": "one_instruction_entailed_source_to_target_transition",
        "initial_ordinal": initial,
        "initial_evidence_exact": initial_evidence,
        "transition": {
            "source_ordinal": source,
            "target_ordinal": target,
            "instruction_evidence_exact": _evidence(
                transition["instruction_evidence_exact"],
                instruction,
                "transition evidence",
            ),
        },
        "exact_event_count": exact_count,
        "event_count_evidence_exact": count_evidence,
        "final_ordinal": final,
        "final_evidence_exact": final_evidence,
        "unmentioned_event_count_or_order_constraints_added": False,
    }


def _curves(
    value: Any, *, instruction: str, expected_route: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 1:
        raise SemanticCaseError(
            "single-transition scope requires exactly one qualitative curve constraint"
        )
    row = _exact(
        value[0],
        {"trend", "completion", "instruction_evidence_exact"},
        "qualitative_curve_constraints[0]",
    )
    trend, completion = row["trend"], row["completion"]
    if trend not in CURVE_TRENDS or completion not in COMPLETION_CONSTRAINTS:
        raise SemanticCaseError("curve trend or completion is invalid")
    final = expected_route["final_ordinal"]
    target = expected_route["transition"]["target_ordinal"]
    if completion == "full_target" and final not in (None, target):
        raise SemanticCaseError(
            "full-target curve conflicts with constrained final endpoint"
        )
    if completion == "partial_target" and final is not None:
        raise SemanticCaseError(
            "partial-target curve cannot constrain one discrete final endpoint"
        )
    return [
        {
            "transition_index": 0,
            "trend": trend,
            "completion": completion,
            "instruction_evidence_exact": _evidence(
                row["instruction_evidence_exact"], instruction, "curve evidence"
            ),
            "numeric_amount_or_percent_target": None,
        }
    ]


def _metric(value: Any) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "activity_rms_dbfs_threshold",
            "delta_presence_relative_db_threshold",
            "minimum_active_bars",
            "threshold_rationale",
            "human_audibility_claimed",
        },
        "metric_config",
    )
    numbers: list[float] = []
    for key in ("activity_rms_dbfs_threshold", "delta_presence_relative_db_threshold"):
        value_at_key = row[key]
        if (
            isinstance(value_at_key, bool)
            or not isinstance(value_at_key, (int, float))
            or not math.isfinite(float(value_at_key))
            or not -300.0 <= float(value_at_key) <= 0.0
        ):
            raise SemanticCaseError(
                f"{key} must be an explicit finite dB value in [-300,0]"
            )
        numbers.append(float(value_at_key))
    minimum = row["minimum_active_bars"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise SemanticCaseError(
            "minimum_active_bars must be a positive integer"
        )
    if row["human_audibility_claimed"] is not False:
        raise SemanticCaseError(
            "energy delta cannot claim human audibility"
        )
    return {
        "activity_rms_dbfs_threshold": numbers[0],
        "delta_presence_relative_db_threshold": numbers[1],
        "minimum_active_bars": int(minimum),
        "threshold_rationale": _text(
            row["threshold_rationale"], "threshold rationale"
        ),
        "human_audibility_claimed": False,
        "metric_scope": "energy_weighted_automation_vs_dry_delta_proxy",
        "barwise_full_history_audio": True,
        "threshold_freeze_scope": "prospective_for_this_bound_v4_run_only",
        "designed_after_prior_results": True,
        "prior_results_used_for_design_but_not_bound_as_evidence": True,
        "independent_confirmation_claimed": False,
    }


def _annotation(
    value: Any,
    *,
    evidence_class: str,
    instruction: str,
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "annotator_count",
            "agreement_frozen_before_execution",
            "direct_cue_reviewer_ids",
            "direct_cue_review_completed_before_execution",
            "direct_cue_review_conclusion",
            "direct_cue_review_rationale",
            "protocol_note",
        },
        "annotation_protocol",
    )
    count = row["annotator_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise SemanticCaseError("at least two annotators are required")
    if row["agreement_frozen_before_execution"] is not True:
        raise SemanticCaseError(
            "annotation agreement must be frozen before execution"
        )
    reviewer_ids = row["direct_cue_reviewer_ids"]
    if (
        isinstance(reviewer_ids, (str, bytes))
        or not isinstance(reviewer_ids, Sequence)
        or len(reviewer_ids) != count
    ):
        raise SemanticCaseError(
            "direct-cue reviewer IDs must identify every annotator"
        )
    reviewers = [
        _text(value, f"direct-cue reviewer ID {index}", maximum=128)
        for index, value in enumerate(reviewer_ids, start=1)
    ]
    if len(reviewers) != len(set(reviewers)):
        raise SemanticCaseError(
            "direct-cue reviewer IDs must be distinct"
        )
    if row["direct_cue_review_completed_before_execution"] is not True:
        raise SemanticCaseError(
            "manual direct-cue review must finish before execution preparation"
        )
    expected_conclusion = (
        "no_unlisted_direct_selector_cue_found"
        if evidence_class == "lexically_disjoint_semantic_paraphrase"
        else "smoke_only_direct_cue_overlap_or_design_blocker"
    )
    if row["direct_cue_review_conclusion"] != expected_conclusion:
        raise SemanticCaseError(
            "manual direct-cue conclusion disagrees with the evidence class"
        )
    return {
        "annotator_count": int(count),
        "agreement_frozen_before_execution": True,
        "direct_cue_reviewer_ids": reviewers,
        "direct_cue_review_completed_before_execution": True,
        "direct_cue_review_conclusion": expected_conclusion,
        "direct_cue_review_rationale": _text(
            row["direct_cue_review_rationale"],
            "manual direct-cue review rationale",
        ),
        "reviewed_automation_instruction_utf8_sha256": _sha_text(instruction),
        "reviewed_endpoint_caller_texts_sha256": _sha_json(
            [endpoint["caller_text_exact"] for endpoint in endpoints]
        ),
        "protocol_note": _text(row["protocol_note"], "annotation protocol note"),
    }


def _classification(
    *, instruction: str, endpoints: Sequence[Mapping[str, Any]], route: Mapping[str, Any]
) -> dict[str, Any]:
    canonical_instruction = _canonical_text(instruction)
    instruction_tokens = set(_lexical_tokens(instruction))
    raw_overlap: list[int] = []
    canonical_overlap: list[int] = []
    lexical_overlap: list[dict[str, Any]] = []
    for endpoint in endpoints:
        text = endpoint["caller_text_exact"]
        if not isinstance(text, str):
            continue
        ordinal = int(endpoint["ordinal"])
        if text in instruction:
            raw_overlap.append(ordinal)
        if _canonical_text(text) in canonical_instruction:
            canonical_overlap.append(ordinal)
        endpoint_tokens = _lexical_tokens(text)
        shared_tokens = sorted(instruction_tokens.intersection(endpoint_tokens))
        lexical_overlap.append(
            {
                "ordinal": ordinal,
                "endpoint_tokens_length_at_least_2": endpoint_tokens,
                "shared_instruction_tokens": shared_tokens,
            }
        )
    direct_cues = [match.group(0) for match in _DIRECT_ORDINAL_CUE.finditer(instruction)]
    direct_cues.extend(match.group(0) for match in _INTERNAL_ID.finditer(instruction))
    route_ordinals = {
        int(route["transition"]["source_ordinal"]),
        int(route["transition"]["target_ordinal"]),
    }
    route_has_reference = any(
        ordinal > 0 and endpoints[ordinal - 1]["kind"] == "processed_reference"
        for ordinal in route_ordinals
    )
    reference_deictics = (
        [match.group(0) for match in _REFERENCE_DEICTIC.finditer(instruction)]
        if route_has_reference
        else []
    )
    natural_language_count = sum(
        endpoint["kind"] == "natural_language" for endpoint in endpoints
    )
    distinct_natural_language_texts = {
        _canonical_text(str(endpoint["caller_text_exact"]))
        for endpoint in endpoints
        if endpoint["kind"] == "natural_language"
    }
    distinct_natural_language_count = len(distinct_natural_language_texts)
    target_ordinal = int(route["transition"]["target_ordinal"])
    target_is_natural_language = bool(
        target_ordinal > 0
        and endpoints[target_ordinal - 1]["kind"] == "natural_language"
    )
    selection_design_blockers: list[str] = []
    if route_has_reference:
        selection_design_blockers.append("processed_reference_on_expected_route")
    if not target_is_natural_language:
        selection_design_blockers.append("expected_target_has_no_caller_semantic_text")
    if natural_language_count < 2:
        selection_design_blockers.append("fewer_than_two_natural_language_alternatives")
    if distinct_natural_language_count < 2:
        selection_design_blockers.append(
            "fewer_than_two_distinct_natural_language_alternatives"
        )
    automatic = (
        "overlap_or_direct_cue_smoke"
        if raw_overlap
        or canonical_overlap
        or any(row["shared_instruction_tokens"] for row in lexical_overlap)
        or direct_cues
        or reference_deictics
        or selection_design_blockers
        else "lexically_disjoint_semantic_paraphrase"
    )
    return {
        "automatic_evidence_class": automatic,
        "raw_case_sensitive_overlap_ordinals_all_natural_language_endpoints": raw_overlap,
        "nfkc_casefold_whitespace_overlap_ordinals_all_natural_language_endpoints": canonical_overlap,
        "zero_shared_token_rule_all_natural_language_endpoints": True,
        "lexical_tokenization": (
            "Unicode NFKC + casefold; language-neutral alphanumeric tokens; "
            "minimum token length 2 codepoints; zero shared tokens required"
        ),
        "instruction_tokens_length_at_least_2": sorted(instruction_tokens),
        "lexical_overlap_all_natural_language_endpoints": lexical_overlap,
        "automatically_detected_direct_ordinal_or_internal_id_cues": direct_cues,
        "route_reference_deictic_cues": reference_deictics,
        "selection_design_blockers": selection_design_blockers,
        "natural_language_alternative_count": natural_language_count,
        "distinct_nfkc_casefold_whitespace_natural_language_alternative_count": (
            distinct_natural_language_count
        ),
        "any_processed_reference_route_is_smoke_only": True,
        "normalization": "Unicode NFKC, casefold, whitespace collapse",
        "transition_source_and_target_both_audited": True,
        "automatic_cue_detector_scope": (
            "conservative English and Korean lexical screen for the reviewed current case"
        ),
        "automatic_cue_detector_exhaustive": False,
        "absence_of_automatic_cue_is_not_sufficient_for_selection_evidence": True,
        "general_multilingual_cue_detection_claimed": False,
    }


def _build_record(
    *,
    case_id: str,
    prepared_path: Path,
    prepared_record: Mapping[str, Any],
    prepared_digest: str,
    evidence_class: str,
    scored_endpoint_ordinals: Sequence[int],
    expected_route: Mapping[str, Any],
    qualitative_curve_constraints: Sequence[Mapping[str, Any]],
    metric_config: Mapping[str, Any],
    annotation_protocol: Mapping[str, Any],
    frozen_at_unix_ns: int,
    case_nonce: str,
) -> dict[str, Any]:
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise SemanticCaseError("case_id is invalid")
    if evidence_class not in EVIDENCE_CLASSES:
        raise SemanticCaseError("evidence_class is invalid")
    if isinstance(frozen_at_unix_ns, bool) or not isinstance(frozen_at_unix_ns, int) or frozen_at_unix_ns < 1:
        raise SemanticCaseError("freeze timestamp is invalid")
    nonce = _text(case_nonce, "case nonce", maximum=256)
    if len(nonce) < 24:
        raise SemanticCaseError("case nonce is too short")
    canonical = prepared_record.get("canonical_manifest")
    if not isinstance(canonical, Mapping):
        raise SemanticCaseError("verified v3 canonical manifest disappeared")
    instruction = _text(canonical.get("automation_instruction"), "automation instruction")
    endpoints = _endpoints(canonical)
    route = _route(expected_route, instruction=instruction, endpoint_count=len(endpoints))
    if isinstance(scored_endpoint_ordinals, (str, bytes)) or not scored_endpoint_ordinals:
        raise SemanticCaseError("scored endpoint ordinals are required")
    scored: list[int] = []
    for raw in scored_endpoint_ordinals:
        ordinal = _ordinal(raw, "scored endpoint ordinal", len(endpoints))
        if ordinal == 0:
            raise SemanticCaseError("dry cannot be a scored semantic endpoint")
        scored.append(ordinal)
    if len(scored) != len(set(scored)):
        raise SemanticCaseError("scored endpoint ordinals must be unique")
    if route["transition"]["target_ordinal"] not in scored:
        raise SemanticCaseError(
            "expected transition target must be included in scored endpoints"
        )
    classification = _classification(
        instruction=instruction, endpoints=endpoints, route=route
    )
    annotation = _annotation(
        annotation_protocol,
        evidence_class=evidence_class,
        instruction=instruction,
        endpoints=endpoints,
    )
    manual_review_class = (
        "lexically_disjoint_semantic_paraphrase"
        if annotation["direct_cue_review_conclusion"]
        == "no_unlisted_direct_selector_cue_found"
        else "overlap_or_direct_cue_smoke"
    )
    automatic_class = classification["automatic_evidence_class"]
    final_class = (
        "overlap_or_direct_cue_smoke"
        if automatic_class == "overlap_or_direct_cue_smoke"
        or manual_review_class == "overlap_or_direct_cue_smoke"
        else "lexically_disjoint_semantic_paraphrase"
    )
    if final_class != evidence_class:
        raise SemanticCaseError(
            "declared evidence class attempts to upgrade or disagrees with the "
            "conservative automatic-plus-manual cue audit"
        )
    classification = {
        **classification,
        "manual_review_evidence_class": manual_review_class,
        "final_evidence_class": final_class,
        "manual_review_downgraded_automatic_paraphrase": bool(
            automatic_class == "lexically_disjoint_semantic_paraphrase"
            and final_class == "overlap_or_direct_cue_smoke"
        ),
        "one_way_policy_automatic_smoke_can_never_upgrade": True,
    }
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "status": STATUS,
        "case_id": case_id,
        "case_nonce": nonce,
        "frozen_at_unix_ns": frozen_at_unix_ns,
        "frozen_at_utc": datetime.fromtimestamp(
            frozen_at_unix_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat(),
        "written_before_execution_pre_registration_and_any_api_call": True,
        "evaluator_code": _identity(Path(__file__)),
        "source_binding": {
            "v3_prepared_directory": str(prepared_path.resolve()),
            "v3_pre_registration_sha256": prepared_digest,
            "canonical_manifest_sha256": _sha_json(canonical),
        },
        "automation_instruction_exact": instruction,
        "automation_instruction_utf8_sha256": _sha_text(instruction),
        "endpoints": endpoints,
        "scored_endpoint_ordinals": scored,
        "evidence": classification,
        "selection_capability_claim_policy": {
            "live_plan_source_and_paraphrase_required": True,
            "overlap_direct_cue_or_reference_deictic_is_smoke_only": True,
            "test_double_is_never_semantic_evidence": True,
            "generalization_claim_allowed": False,
        },
        "annotation_protocol": annotation,
        "expected_route": route,
        "qualitative_curve_constraints": _curves(
            qualitative_curve_constraints,
            instruction=instruction,
            expected_route=route,
        ),
        "metric_config": _metric(metric_config),
        "visibility_contract": {
            "caller_instruction_and_endpoint_texts_remain_planner_visible": True,
            "evaluator_only_route_curve_threshold_annotation_nonce_hidden": True,
            "expected_route_sent_to_any_llm": False,
            "curve_constraints_sent_to_any_llm": False,
            "metric_thresholds_sent_to_any_llm": False,
            "annotation_protocol_sent_to_any_llm": False,
            "case_nonce_sent_to_any_llm": False,
        },
        "preparation_work": {
            "api_calls": 0,
            "audio_renders": 0,
            "optimizer_runs": 0,
            "optimizer_steps": 0,
            "encoder_loads": 0,
            "audio_files_written": 0,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SemanticCaseError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticCaseError(f"expected JSON object: {path}")
    return value


def prepare_case(
    *,
    case_id: str,
    prepared_path: Path,
    output: Path,
    evidence_class: str,
    scored_endpoint_ordinals: Sequence[int],
    expected_route: Mapping[str, Any],
    qualitative_curve_constraints: Sequence[Mapping[str, Any]],
    metric_config: Mapping[str, Any],
    annotation_protocol: Mapping[str, Any],
) -> Path:
    output = output.resolve()
    if output.exists():
        raise SemanticCaseError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging_", dir=output.parent))
    try:
        prepared_record, prepared_digest = prepare.verify_prepared(prepared_path)
        record = _build_record(
            case_id=case_id,
            prepared_path=prepared_path,
            prepared_record=prepared_record,
            prepared_digest=prepared_digest,
            evidence_class=evidence_class,
            scored_endpoint_ordinals=scored_endpoint_ordinals,
            expected_route=expected_route,
            qualitative_curve_constraints=qualitative_curve_constraints,
            metric_config=metric_config,
            annotation_protocol=annotation_protocol,
            frozen_at_unix_ns=time.time_ns(),
            case_nonce=secrets.token_hex(24),
        )
        prereg = staging / "SEMANTIC_PRE_REGISTRATION.json"
        _write_json(prereg, record)
        digest = manifest.file_sha256(prereg)
        (staging / "SEMANTIC_PRE_REGISTRATION.sha256").write_text(
            f"{digest}  SEMANTIC_PRE_REGISTRATION.json\n", encoding="ascii"
        )
        _write_json(
            staging / "PREPARATION_STATUS.json",
            {
                "status": STATUS,
                "evaluator_version": EVALUATOR_VERSION,
                "semantic_pre_registration_sha256": digest,
                "sealed_at_unix_ns": time.time_ns(),
                "api_calls": 0,
                "audio_renders": 0,
                "optimizer_runs": 0,
                "optimizer_steps": 0,
                "encoder_loads": 0,
                "audio_files_written": 0,
                "review_required_before_execution_preparation": True,
            },
        )
    except Exception as error:
        _write_json(
            staging / "FAILURE.json",
            {
                "status": "semantic_case_prepare_failed",
                "evaluator_version": EVALUATOR_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "api_calls": 0,
                "audio_renders": 0,
                "optimizer_runs": 0,
                "traceback": traceback.format_exc(),
            },
        )
        os.replace(staging, output)
        raise
    os.replace(staging, output)
    return output


def verify_prepared_case(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    path = path.resolve()
    allowed = {
        "SEMANTIC_PRE_REGISTRATION.json",
        "SEMANTIC_PRE_REGISTRATION.sha256",
        "PREPARATION_STATUS.json",
    }
    if not path.is_dir() or {item.name for item in path.iterdir()} != allowed:
        raise SemanticCaseError("semantic preregistration file set changed")
    prereg_path = path / "SEMANTIC_PRE_REGISTRATION.json"
    digest = manifest.file_sha256(prereg_path)
    tokens = path.joinpath("SEMANTIC_PRE_REGISTRATION.sha256").read_text(
        encoding="ascii"
    ).split()
    if tokens != [digest, "SEMANTIC_PRE_REGISTRATION.json"]:
        raise SemanticCaseError("semantic preregistration sidecar changed")
    record = _read_json(prereg_path)
    status = _read_json(path / "PREPARATION_STATUS.json")
    sealed_at = status.get("sealed_at_unix_ns")
    expected_status = {
        "status": STATUS,
        "evaluator_version": EVALUATOR_VERSION,
        "semantic_pre_registration_sha256": digest,
        "sealed_at_unix_ns": sealed_at,
        "api_calls": 0,
        "audio_renders": 0,
        "optimizer_runs": 0,
        "optimizer_steps": 0,
        "encoder_loads": 0,
        "audio_files_written": 0,
        "review_required_before_execution_preparation": True,
    }
    if (
        status != expected_status
        or isinstance(sealed_at, bool)
        or not isinstance(sealed_at, int)
        or sealed_at < int(record.get("frozen_at_unix_ns", 0))
    ):
        raise SemanticCaseError("semantic preparation status changed")
    if record.get("evaluator_code") != _identity(Path(__file__)):
        raise SemanticCaseError("semantic prereg evaluator code changed")
    binding = record.get("source_binding")
    if not isinstance(binding, Mapping):
        raise SemanticCaseError("semantic source binding disappeared")
    prepared_dir = Path(binding["v3_prepared_directory"])
    prepared_record, prepared_digest = prepare.verify_prepared(prepared_dir)
    if binding.get("v3_pre_registration_sha256") != prepared_digest:
        raise SemanticCaseError("bound v3 preparation digest changed")
    rebuilt = _build_record(
        case_id=record.get("case_id"),
        prepared_path=prepared_dir,
        prepared_record=prepared_record,
        prepared_digest=prepared_digest,
        evidence_class=record.get("evidence", {}).get("final_evidence_class"),
        scored_endpoint_ordinals=record.get("scored_endpoint_ordinals", []),
        expected_route={
            key: record.get("expected_route", {}).get(key)
            for key in (
                "initial_ordinal",
                "initial_evidence_exact",
                "transition",
                "exact_event_count",
                "event_count_evidence_exact",
                "final_ordinal",
                "final_evidence_exact",
            )
        },
        qualitative_curve_constraints=[
            {
                "trend": row.get("trend"),
                "completion": row.get("completion"),
                "instruction_evidence_exact": row.get("instruction_evidence_exact"),
            }
            for row in record.get("qualitative_curve_constraints", [])
        ],
        metric_config={
            key: record.get("metric_config", {}).get(key)
            for key in (
                "activity_rms_dbfs_threshold",
                "delta_presence_relative_db_threshold",
                "minimum_active_bars",
                "threshold_rationale",
                "human_audibility_claimed",
            )
        },
        annotation_protocol={
            key: record.get("annotation_protocol", {}).get(key)
            for key in (
                "annotator_count",
                "agreement_frozen_before_execution",
                "direct_cue_reviewer_ids",
                "direct_cue_review_completed_before_execution",
                "direct_cue_review_conclusion",
                "direct_cue_review_rationale",
                "protocol_note",
            )
        },
        frozen_at_unix_ns=record.get("frozen_at_unix_ns"),
        case_nonce=record.get("case_nonce"),
    )
    if rebuilt != record:
        raise SemanticCaseError(
            "semantic preregistration changed or contains unknown fields"
        )
    return record, digest, status
