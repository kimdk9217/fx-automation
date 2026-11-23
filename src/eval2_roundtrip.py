"""Evaluation 2 (curve reconstruction): prepare, run, score and verify the 90 instruction cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import llm
import planner
import reference_bridge as bridge
import reference_matcher
import roundtrip_common as roundtrip

EXPECTED_GROUND_TRUTH_SHA256 = "53e87b395d850348779f5278a143c951d46ea893248bb0c58480da9252c42c74"
AUTHOR_MODEL = "gpt-5.6-luna"
PLANNER_MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
N_CURVES = 30
DESCRIPTIONS_PER_CURVE = 3
N_CASES = N_CURVES * DESCRIPTIONS_PER_CURVE

if AUTHOR_MODEL == PLANNER_MODEL:
    raise RuntimeError("instruction author and automation planner must differ")


ROOT = Path(__file__).resolve().parent
RUN_VERSION = "synthetic-curve-roundtrip-v4-luna3-terra-all90"
DEFAULT_OUTPUT = ROOT / "demo_outputs" / "synthetic_curve_roundtrip_v4_luna3_terra_all90"
SOURCE = ROOT / "demo_outputs" / "synthetic_curve_roundtrip_v3_luna3_terra"


class RoundTripRunError(RuntimeError):
    pass


def _code_hashes() -> dict[str, dict[str, str]]:
    modules = {
        "planner_validation": Path(planner.__file__).resolve(),
        "planner_parameter_contract": Path(bridge.__file__).resolve(),
        "planner_bridge_base": Path(bridge.__file__).resolve(),
        "planner_parameter_counts": Path(reference_matcher.__file__).resolve(),
        "base_scoring": Path(roundtrip.__file__).resolve(),
        "corrected_prompt": Path(roundtrip.__file__).resolve(),
        "v3_constants": Path(__file__).resolve(),
        "api_client": Path(llm.__file__).resolve(),
        "api_schema_shim": Path(llm.__file__).resolve(),
        "v4_runner": Path(__file__).resolve(),
    }
    return {
        name: {"path": str(path), "sha256": _sha(path)}
        for name, path in modules.items()
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_or_match(path: Path, value: Any) -> None:
    if path.exists():
        if _read(path) != value:
            raise RoundTripRunError(f"existing artifact changed: {path}")
        return
    _write(path, value)


def _sha(path: Path) -> str:
    return roundtrip._sha_file(path)


def _source_material() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_path = SOURCE / "GROUND_TRUTH.json"
    requests_path = SOURCE / "AUTHOR_REQUESTS.json"
    completed_path = SOURCE / "AUTHOR_COMPLETED.json"
    for path in (gt_path, requests_path, completed_path):
        if not path.is_file():
            raise RoundTripRunError(f"missing frozen source artifact: {path}")
    if _sha(gt_path) != EXPECTED_GROUND_TRUTH_SHA256:
        raise RoundTripRunError("source ground truth changed")
    ground_truth = _read(gt_path)
    author_requests = _read(requests_path)
    author_completed = _read(completed_path)
    if len(ground_truth) != N_CURVES or len(author_requests) != N_CURVES:
        raise RoundTripRunError("source curve or author-request count changed")
    request_by_curve = {row["curve_id"]: row for row in author_requests}
    terminal_by_curve = {row["curve_id"]: row for row in author_completed["terminals"]}
    cases: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = [
        {"kind": "ground_truth", "path": str(gt_path.resolve()), "sha256": _sha(gt_path)},
        {"kind": "author_requests", "path": str(requests_path.resolve()), "sha256": _sha(requests_path)},
        {"kind": "author_completed", "path": str(completed_path.resolve()), "sha256": _sha(completed_path)},
    ]
    audit = []
    for oracle in ground_truth:
        curve_id = oracle["curve_id"]
        request = request_by_curve[curve_id]
        directory = SOURCE / "author" / curve_id
        request_path = directory / "request.json"
        response_path = directory / "raw_response_before_parse.json"
        raw_path = directory / "raw_author_output.json"
        for path in (request_path, response_path, raw_path):
            if not path.is_file():
                raise RoundTripRunError(f"missing author source file: {path}")
        if _read(request_path) != request:
            raise RoundTripRunError(f"author request changed: {curve_id}")
        response_record = _read(response_path)
        if response_record["native_request"] != request["native_request"]:
            raise RoundTripRunError(f"author native request changed: {curve_id}")
        response = response_record["response"]
        roundtrip._assert_response_model(response, AUTHOR_MODEL)
        raw = roundtrip._raw_content(response)
        if _read(raw_path) != raw:
            raise RoundTripRunError(f"raw author output changed: {curve_id}")
        instructions = raw.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != DESCRIPTIONS_PER_CURVE:
            raise RoundTripRunError(f"author must provide three descriptions: {curve_id}")
        if not all(isinstance(value, str) and value.strip() for value in instructions):
            raise RoundTripRunError(f"author description is not text: {curve_id}")
        files.extend(
            {
                "kind": kind,
                "curve_id": curve_id,
                "path": str(path.resolve()),
                "sha256": _sha(path),
            }
            for kind, path in (
                ("author_request", request_path),
                ("author_response", response_path),
                ("author_output", raw_path),
            )
        )
        terminal = terminal_by_curve[curve_id]
        for index, instruction in enumerate(instructions, start=1):
            try:
                roundtrip.validate_natural_instruction(instruction)
            except Exception as error:
                passes = False
                reason = str(error)
            else:
                passes = True
                reason = None
            case_id = f"{curve_id}_p{index}"
            audit.append({"case_id": case_id, "passes_lexical_audit": passes, "reason": reason})
            cases.append(
                {
                    "case_id": case_id,
                    "curve_id": curve_id,
                    "family": oracle["family"],
                    "paraphrase_index": index,
                    "instruction": instruction,
                    "instruction_utf8_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                    "source_author_terminal_valid": bool(terminal["valid"]),
                    "passes_descriptive_lexical_audit": passes,
                    "lexical_audit_reason": reason,
                }
            )
    if len(cases) != N_CASES or len({row["case_id"] for row in cases}) != N_CASES:
        raise RoundTripRunError("fixed ninety-case source denominator changed")
    manifest = {
        "source_output": str(SOURCE.resolve()),
        "files": files,
        "descriptions": N_CASES,
        "selection": 0,
        "editing": 0,
        "author_retries": 0,
        "v4_author_api_calls": 0,
        "instruction_order": [row["case_id"] for row in cases],
        "instruction_hashes": {row["case_id"]: row["instruction_utf8_sha256"] for row in cases},
        "lexical_audit_is_descriptive_only": True,
        "lexical_audit": audit,
    }
    return ground_truth, cases, manifest


def _request(case: Mapping[str, Any]) -> dict[str, Any]:
    payload = roundtrip._planner_payload(str(case["instruction"]))
    if payload["model"] != PLANNER_MODEL:
        raise RoundTripRunError("planner model changed")
    return {
        **deepcopy(dict(case)),
        "context": roundtrip._planner_context(str(case["instruction"])),
        "local_schema": roundtrip._planner_schema(),
        "payload": payload,
        "native_request": roundtrip._native_request(payload, REASONING_EFFORT),
    }


def prepare(output: Path) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RoundTripRunError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ground_truth, cases, manifest = _source_material()
    (output / "GROUND_TRUTH.json").write_bytes((SOURCE / "GROUND_TRUTH.json").read_bytes())
    requests = [_request(case) for case in cases]
    _write(output / "SOURCE_MANIFEST.json", manifest)
    _write(output / "PLANNER_CASES.json", cases)
    _write(output / "PLANNER_REQUESTS.json", requests)
    prereg = {
        "run_version": RUN_VERSION,
        "status": "frozen_before_any_v4_planner_call",
        "role": "replacement_primary_round_trip_evaluation",
        "scope": {
            "ground_truth_curves": N_CURVES,
            "families": 6,
            "descriptions_per_curve": DESCRIPTIONS_PER_CURVE,
            "fixed_planner_cases": N_CASES,
            "planner_calls": N_CASES,
        },
        "models": {
            "instruction_author": AUTHOR_MODEL,
            "automation_planner": PLANNER_MODEL,
            "reasoning_effort": REASONING_EFFORT,
        },
        "protocol": {
            "all_ninety_descriptions_reused_byte_for_byte": True,
            "lexical_audit_is_not_an_execution_gate": True,
            "retry": 0,
            "repair": 0,
            "editing": 0,
            "selection": 0,
            "invalid_plan_penalty": 1.0,
            "corrected_generic_blend_contract": True,
        },
        "metrics": {
            "independent_unit": "ground-truth curve; three descriptions averaged before n=30 aggregation",
            "primary": "curve MAE over 1001 points",
            "secondary": ["event sequence exact", "change-point error", "state recovery", "pairwise reconstruction MAE", "oracle rank"],
            "bootstrap_replicates": roundtrip.N_BOOTSTRAP,
            "bootstrap_seed": roundtrip.BOOTSTRAP_SEED,
            "stratified_by_family": True,
        },
        "source_manifest": manifest,
        "frozen": {
            "ground_truth_sha256": _sha(output / "GROUND_TRUTH.json"),
            "source_manifest_sha256": _sha(output / "SOURCE_MANIFEST.json"),
            "planner_cases_sha256": _sha(output / "PLANNER_CASES.json"),
            "planner_requests_sha256": _sha(output / "PLANNER_REQUESTS.json"),
            "planner_schema_sha256": roundtrip._sha_json(roundtrip._planner_schema()),
            "planner_prompt_sha256": hashlib.sha256(roundtrip.PLANNER_SYSTEM_PROMPT.encode()).hexdigest(),
            "code": _code_hashes(),
        },
    }
    _write(output / "PRE_REGISTRATION.json", prereg)
    _write(
        output / "PREPARED.json",
        {
            "run_version": RUN_VERSION,
            "ground_truth_sha256": _sha(output / "GROUND_TRUTH.json"),
            "source_manifest_sha256": _sha(output / "SOURCE_MANIFEST.json"),
            "planner_cases_sha256": _sha(output / "PLANNER_CASES.json"),
            "planner_requests_sha256": _sha(output / "PLANNER_REQUESTS.json"),
            "pre_registration_sha256": _sha(output / "PRE_REGISTRATION.json"),
            "planner_api_calls": 0,
        },
    )
    print("prepared v4: 30 curves, 90 frozen descriptions, 90 Terra planner calls")


def _verify_prepared(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ground_truth, cases, manifest = _source_material()
    requests = [_request(case) for case in cases]
    if _read(output / "GROUND_TRUTH.json") != ground_truth:
        raise RoundTripRunError("ground truth changed")
    if _read(output / "SOURCE_MANIFEST.json") != manifest:
        raise RoundTripRunError("source manifest changed")
    if _read(output / "PLANNER_CASES.json") != cases or _read(output / "PLANNER_REQUESTS.json") != requests:
        raise RoundTripRunError("planner cases or requests changed")
    prepared = _read(output / "PREPARED.json")
    prereg = _read(output / "PRE_REGISTRATION.json")
    for filename, key in (
        ("GROUND_TRUTH.json", "ground_truth_sha256"),
        ("SOURCE_MANIFEST.json", "source_manifest_sha256"),
        ("PLANNER_CASES.json", "planner_cases_sha256"),
        ("PLANNER_REQUESTS.json", "planner_requests_sha256"),
        ("PRE_REGISTRATION.json", "pre_registration_sha256"),
    ):
        if _sha(output / filename) != prepared[key]:
            raise RoundTripRunError(f"prepared hash changed: {filename}")
    if prereg["frozen"]["code"] != _code_hashes():
        raise RoundTripRunError("evaluation code changed after freeze")
    return ground_truth, cases, requests


def _failure(case_id: str, error: BaseException | str) -> dict[str, Any]:
    return {
        "kind": "planner",
        "identifier": case_id,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "fixed_failure",
        "error": str(error),
        "traceback": traceback.format_exc() if isinstance(error, BaseException) else None,
        "retry": 0,
        "repair": 0,
        "selection": 0,
    }


def _finish(case: Mapping[str, Any], request: Mapping[str, Any], directory: Path, record: Mapping[str, Any], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    if record["native_request"] != request["native_request"]:
        raise RoundTripRunError("native request changed")
    response = record["response"]
    roundtrip._assert_response_model(response, PLANNER_MODEL)
    raw = roundtrip._raw_content(response)
    _write_or_match(directory / "raw_plan.json", raw)
    validated = planner.validate_plan(raw, schema=request["local_schema"], registry=roundtrip._Registry(), instruction=str(case["instruction"]))
    _write_or_match(directory / "validated_plan.json", validated)
    return roundtrip.score_prediction(case, validated, ground_truth)


def execute(output: Path) -> None:
    output = output.resolve()
    ground_truth, cases, requests = _verify_prepared(output)
    forbidden = [
        output / "planner",
        output / "PLANNER_COMPLETED.json",
        output / "RESULTS.json",
        output / "VERIFIED.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RoundTripRunError(
            "planner output is not pristine; this frozen run does not resume or retry partial calls"
        )
    llm.api_key()
    scores = []
    for case, request in zip(cases, requests):
        case_id = case["case_id"]
        directory = output / "planner" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        _write_or_match(directory / "request.json", request)
        started = directory / "CALL_STARTED.json"
        response_path = directory / "raw_response_before_parse.json"
        score_path = directory / "score.json"
        _write(started, {"case_id": case_id, "attempt": 1})
        try:
            response, native = llm.call_openai_structured_once(
                request["payload"], base_url=llm.DEFAULT_BASE_URL,
                timeout=900.0, reasoning_effort=REASONING_EFFORT,
            )
            record = {"response": response, "native_request": native}
            _write(response_path, record)
            case_score = _finish(case, request, directory, record, ground_truth)
        except Exception as error:
            case_score = roundtrip.invalid_score(case, _failure(case_id, error))
        _write(score_path, case_score)
        terminal = {
            "case_id": case_id,
            "valid_plan": bool(case_score["valid_plan"]),
            "failure": case_score["failure"],
            "score_sha256": _sha(score_path),
        }
        _write(directory / ("COMPLETED.json" if case_score["valid_plan"] else "FAILURE.json"), terminal)
        scores.append(case_score)
    if len(scores) != N_CASES:
        raise RoundTripRunError("ninety-case denominator changed")
    attempts = sum((output / "planner" / case["case_id"] / "CALL_STARTED.json").exists() for case in cases)
    stored_scores = sum((output / "planner" / case["case_id"] / "score.json").exists() for case in cases)
    if attempts != N_CASES or stored_scores != N_CASES:
        raise RoundTripRunError("not all ninety fixed planner attempts reached a terminal score")
    _write(
        output / "PLANNER_COMPLETED.json",
        {
            "run_version": RUN_VERSION,
            "planner_attempts": attempts,
            "fixed_denominator": N_CASES,
            "valid_plans": sum(bool(row["valid_plan"]) for row in scores),
            "invalid_plans": sum(not bool(row["valid_plan"]) for row in scores),
        },
    )
    score(output, scores)


def _curve_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault(row["curve_id"], []).append(row)
    if len(grouped) != N_CURVES or any(len(rows) != DESCRIPTIONS_PER_CURVE for rows in grouped.values()):
        raise RoundTripRunError("each curve must have exactly three fixed descriptions")
    result = []
    for curve_id in sorted(grouped):
        rows = sorted(grouped[curve_id], key=lambda row: row["paraphrase_index"])
        predictions = [
            None if row["predicted_curve_1001"] is None else np.asarray(row["predicted_curve_1001"], dtype=np.float64)
            for row in rows
        ]
        pairwise_values = []
        for left in range(DESCRIPTIONS_PER_CURVE):
            for right in range(left + 1, DESCRIPTIONS_PER_CURVE):
                if predictions[left] is None or predictions[right] is None:
                    pairwise_values.append(1.0)
                else:
                    pairwise_values.append(float(np.mean(np.abs(predictions[left] - predictions[right]))))
        result.append(
            {
                "curve_id": curve_id,
                "family": rows[0]["family"],
                "event_sequence_exact_rate": float(np.mean([row["metrics"]["event_sequence_exact"] for row in rows])),
                "curve_mae": float(np.mean([row["metrics"]["curve_mae"] for row in rows])),
                "penalized_change_point_error": float(np.mean([row["metrics"]["penalized_change_point_error"] for row in rows])),
                "initial_state_accuracy": float(np.mean([row["metrics"]["initial_state_exact"] for row in rows])),
                "final_state_absolute_error": float(np.mean([row["metrics"]["final_state_absolute_error"] for row in rows])),
                "final_state_within_0_01_rate": float(np.mean([row["metrics"]["final_state_within_0_01"] for row in rows])),
                "correct_oracle_top1_rate": float(np.mean([row["metrics"]["correct_oracle_top1"] for row in rows])),
                "correct_oracle_mean_rank": float(np.mean([row["metrics"]["correct_oracle_rank"] for row in rows])),
                "paraphrase_mae": float(np.mean(pairwise_values)),
            }
        )
    return result


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scores) != N_CASES:
        raise RoundTripRunError("aggregate requires ninety fixed cases")
    curve_rows = _curve_rows(scores)
    exact_count = sum(bool(row["metrics"]["event_sequence_exact"]) for row in scores)
    return {
        "run_version": RUN_VERSION,
        "condition": "all_90_luna_descriptions_to_terra_planner",
        "fixed_denominator": {
            "independent_curves": N_CURVES,
            "planner_cases": N_CASES,
            "descriptions_per_curve": DESCRIPTIONS_PER_CURVE,
        },
        "valid_plans": sum(bool(row["valid_plan"]) for row in scores),
        "invalid_plans": sum(not bool(row["valid_plan"]) for row in scores),
        "case_level_event_sequence_exact_count": exact_count,
        "case_level_event_sequence_exact_rate": exact_count / N_CASES,
        "primary_metric": "curve_mae",
        "aggregate_curve_level_stratified_bootstrap_10000": roundtrip.stratified_bootstrap(curve_rows),
        "curve_level_rows": curve_rows,
        "case_scores": scores,
        "interpretation_limits": [
        "Instructions are model-authored from synthetic curves, not human-authored.",
        "This measures the combined author-plus-planner round-trip channel.",
        "All ninety descriptions are included without lexical filtering, editing, repair, retry, or selection.",
        "Three descriptions of one oracle are averaged before n=30 aggregation.",
        "No audio rendering, reference matching, perceptual quality, or musical optimality is evaluated.",
        "Bootstrap intervals describe this fixed synthetic set, not a population confidence interval.",
        ],
    }


def _verify_case_chain(
    output: Path,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    directory = output / "planner" / case_id
    required = (directory / "request.json", directory / "CALL_STARTED.json", directory / "score.json")
    if not all(path.is_file() for path in required):
        raise RoundTripRunError(f"incomplete fixed case chain: {case_id}")
    if _read(directory / "request.json") != request:
        raise RoundTripRunError(f"stored planner request changed: {case_id}")
    if _read(directory / "CALL_STARTED.json") != {"case_id": case_id, "attempt": 1}:
        raise RoundTripRunError(f"planner attempt marker changed: {case_id}")
    stored = _read(directory / "score.json")
    response_path = directory / "raw_response_before_parse.json"
    raw_path = directory / "raw_plan.json"
    validated_path = directory / "validated_plan.json"
    if response_path.exists():
        record = _read(response_path)
        if record.get("native_request") != request["native_request"]:
            raise RoundTripRunError(f"native planner request changed: {case_id}")
        try:
            response = record["response"]
            roundtrip._assert_response_model(response, PLANNER_MODEL)
            raw = roundtrip._raw_content(response)
        except Exception:
            if raw_path.exists() or validated_path.exists():
                raise RoundTripRunError(f"unexpected parsed artifact after response failure: {case_id}")
            expected = roundtrip.invalid_score(case, deepcopy(stored.get("failure")))
        else:
            if not raw_path.is_file() or _read(raw_path) != raw:
                raise RoundTripRunError(f"raw planner output changed or missing: {case_id}")
            try:
                validated = planner.validate_plan(
                    raw,
                    schema=request["local_schema"],
                    registry=roundtrip._Registry(),
                    instruction=str(case["instruction"]),
                )
            except Exception:
                if validated_path.exists():
                    raise RoundTripRunError(f"unexpected validated plan after validation failure: {case_id}")
                expected = roundtrip.invalid_score(case, deepcopy(stored.get("failure")))
            else:
                if not validated_path.is_file() or _read(validated_path) != validated:
                    raise RoundTripRunError(f"validated plan changed or missing: {case_id}")
                expected = roundtrip.score_prediction(case, validated, ground_truth)
    else:
        if raw_path.exists() or validated_path.exists():
            raise RoundTripRunError(f"parsed artifact exists without API response: {case_id}")
        expected = roundtrip.invalid_score(case, deepcopy(stored.get("failure")))
    if stored != expected:
        raise RoundTripRunError(f"stored score differs from replay: {case_id}")
    completed_path = directory / "COMPLETED.json"
    failure_path = directory / "FAILURE.json"
    if completed_path.exists() == failure_path.exists():
        raise RoundTripRunError(f"planner terminal state is ambiguous: {case_id}")
    terminal_path = completed_path if stored["valid_plan"] else failure_path
    expected_terminal = {
        "case_id": case_id,
        "valid_plan": bool(stored["valid_plan"]),
        "failure": stored["failure"],
        "score_sha256": _sha(directory / "score.json"),
    }
    if not terminal_path.is_file() or _read(terminal_path) != expected_terminal:
        raise RoundTripRunError(f"planner terminal record changed: {case_id}")
    return stored


def _verify_execution(output: Path) -> list[dict[str, Any]]:
    ground_truth, cases, requests = _verify_prepared(output)
    completed_path = output / "PLANNER_COMPLETED.json"
    if not completed_path.is_file():
        raise RoundTripRunError("missing planner completion record")
    expected_ids = {str(case["case_id"]) for case in cases}
    planner_root = output / "planner"
    if not planner_root.is_dir() or {path.name for path in planner_root.iterdir()} != expected_ids:
        raise RoundTripRunError("planner directory set differs from the fixed ninety cases")
    scores = [
        _verify_case_chain(output, case, request, ground_truth)
        for case, request in zip(cases, requests)
    ]
    expected_completed = {
        "run_version": RUN_VERSION,
        "planner_attempts": N_CASES,
        "fixed_denominator": N_CASES,
        "valid_plans": sum(bool(row["valid_plan"]) for row in scores),
        "invalid_plans": sum(not bool(row["valid_plan"]) for row in scores),
    }
    if _read(completed_path) != expected_completed:
        raise RoundTripRunError("planner completion totals changed")
    return scores


def _scores(output: Path) -> list[dict[str, Any]]:
    return _verify_execution(output)


def score(output: Path, scores: list[dict[str, Any]] | None = None) -> None:
    output = output.resolve()
    result = aggregate(_scores(output) if scores is None else scores)
    _write_or_match(output / "RESULTS.json", result)
    metrics = result["aggregate_curve_level_stratified_bootstrap_10000"]
    print(
        f"scored v4: event exact {metrics['event_sequence_exact_rate']['mean']:.3f}, "
        f"curve MAE {metrics['curve_mae']['mean']:.3f}"
    )


def verify(output: Path) -> None:
    output = output.resolve()
    scores = _verify_execution(output)
    expected = aggregate(scores)
    if _read(output / "RESULTS.json") != expected:
        raise RoundTripRunError("results changed")
    _write_or_match(
        output / "VERIFIED.json",
        {
            "run_version": RUN_VERSION,
            "verified": True,
            "ground_truth_sha256": EXPECTED_GROUND_TRUTH_SHA256,
            "source_descriptions": N_CASES,
            "planner_attempts": N_CASES,
            "results_sha256": _sha(output / "RESULTS.json"),
            "all_ninety_cases_included": True,
        },
    )
    print("verified synthetic curve round-trip v4")


def _parse(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse(argv)
    {"prepare": prepare, "execute": execute, "score": score, "verify": verify}[args.command](args.output)


if __name__ == "__main__":
    main()
