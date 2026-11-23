"""Prepares a run request from the instruction, target stem, section and reference audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping

import endpoint_manifest as manifest
import llm
import planner
import reference_bridge as bridge
import reference_matcher as matcher
import text_endpoints as endpoint_contract

RUN_VERSION = "medleydb-dynamic-multi-endpoint-prepare-v3"
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"
AUTOMATION_SYSTEM_PROMPT = """\
Convert the caller's exact automation instruction into a time-varying plan over
the registered endpoints. Endpoint identifiers are opaque internal labels.
Natural-language descriptions, when present, are exact caller input; they are
not built-in classes. Processed-reference labels are ordinal only.

Choose event count, order, amounts, holds, curve handles, initial state, and
final state only from the caller instruction. No scenario, endpoint sequence,
amount, or final endpoint is supplied elsewhere. Return one schema-valid JSON
object. Do not output endpoint DSP parameters, audio, embeddings, presets,
lookup tables, retries, or repairs.
"""


class PrepareRunError(RuntimeError):
    pass


class _DeferredRegistry:
    def __init__(self, mode: str, endpoint_ids: tuple[str, ...]) -> None:
        self.mode = mode
        self.endpoint_ids = endpoint_ids


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise PrepareRunError(f"missing frozen file: {path}")
    return {"path": str(path), "sha256": manifest.file_sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrepareRunError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _code_identities() -> dict[str, dict[str, str]]:
    modules = {
        "runner": sys.modules[__name__],
        "manifest": manifest,
        "static_endpoint_contract": endpoint_contract,
        "automation_planner_schema": planner,
        "openai_structured_call": llm,
        "reference_matcher": matcher,
        "joint_renderer_contract": bridge,
    }
    return {
        name: _identity(Path(module.__file__)) for name, module in modules.items()
    }


def _automation_declarations(
    canonical: manifest.CanonicalManifest,
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = [
        {
            "endpoint_id": "endpoint_000",
            "identifier_visibility": "hidden_internal_identifier",
            "role": "dry",
            "caller_description_exact": None,
        }
    ]
    reference_index = 0
    for endpoint in canonical.endpoints:
        if endpoint.kind == "processed_reference":
            reference_index += 1
            declarations.append(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "identifier_visibility": "hidden_internal_identifier",
                    "role": "processed_reference",
                    "reference_ordinal": reference_index,
                    "caller_description_exact": None,
                }
            )
        else:
            declarations.append(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "identifier_visibility": "hidden_internal_identifier",
                    "role": "caller_natural_language",
                    "caller_description_exact": endpoint.user_description,
                    "caller_description_utf8_sha256": hashlib.sha256(
                        str(endpoint.user_description).encode("utf-8")
                    ).hexdigest(),
                }
            )
    return declarations


def _automation_schema(canonical: manifest.CanonicalManifest) -> dict[str, Any]:
    endpoint_ids = ("endpoint_000",) + tuple(
        endpoint.endpoint_id for endpoint in canonical.endpoints
    )
    registry = _DeferredRegistry(canonical.renderer_mode, endpoint_ids)
    return planner.make_plan_schema(
        registry=registry,
        allowed_stems=[str(canonical.target["stem"])],
        allowed_sections=[str(canonical.target["section"])],
        allowed_section_occurrences=[
            (str(canonical.target["section"]), int(canonical.target["occurrence"]))
        ],
        locked_stem=str(canonical.target["stem"]),
        locked_section=str(canonical.target["section"]),
        locked_occurrence=int(canonical.target["occurrence"]),
    )


def build_pre_registration(
    *,
    manifest_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    base_url: str = llm.DEFAULT_BASE_URL,
) -> dict[str, Any]:
    canonical = manifest.load_manifest(manifest_path)
    routes = [
        endpoint_contract.build_future_route(
            endpoint,
            target=canonical.target,
            renderer_mode=canonical.renderer_mode,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
        )
        for endpoint in canonical.endpoints
    ]
    schema = _automation_schema(canonical)
    declarations = _automation_declarations(canonical)
    if [row["endpoint_id"] for row in declarations] != [
        "endpoint_000",
        *(endpoint.endpoint_id for endpoint in canonical.endpoints),
    ]:
        raise PrepareRunError("opaque endpoint order changed")
    return {
        "run_version": RUN_VERSION,
        "status": "prepared_not_executed",
        "written_before_any_v3_api_call": True,
        "work_counters": {
            "api_calls": 0,
            "audio_renders": 0,
            "optimizer_runs": 0,
            "optimizer_steps": 0,
            "audio_files_written": 0,
        },
        "public_input_contract": {
            "caller_supplies": [
                "exact automation instruction",
                "exact natural-language endpoint text and explicit effect mode",
                "optional processed-reference audio and explicit effect mode",
                "target stem, section, occurrence, and structure",
            ],
            "caller_supplies_endpoint_id_or_name": False,
            "instruction_only_endpoint_extraction_claimed": False,
            "reason_for_explicit_effect_mode": (
                "processor routing is caller input in v3 and is never inferred from a style word"
            ),
        },
        "canonical_manifest": canonical.to_dict(),
        "endpoint_generation_config": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "base_url": base_url,
        },
        "internal_endpoint_declarations": declarations,
        "endpoint_materialization_routes": routes,
        "future_work_budget": endpoint_contract.route_budget(routes),
        "automation_planner": {
            "system_prompt": AUTOMATION_SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(
                AUTOMATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "dynamic_schema": schema,
            "dynamic_schema_sha256": _sha_json(schema),
            "endpoint_parameters_and_common_target_summaries_materialized": False,
            "live_context_and_native_request_materialized": False,
            "why_deferred": (
                "endpoint DSP parameters and common-target summaries do not exist before "
                "the separately reviewed materialization stage"
            ),
            "event_count_range": [0, planner.MAX_EVENTS],
            "forced_event_count": None,
            "forced_event_order": None,
            "forced_amounts": None,
            "forced_initial_endpoint": None,
            "forced_final_endpoint": None,
            "forced_hold_pattern": None,
        },
        "no_hidden_style_contract": {
            "system_defined_style_labels": [],
            "style_lookup_table": False,
            "style_presets": False,
            "production_style_demonstrations": False,
            "injected_curve_scenario": False,
            "caller_text_forwarded_exactly": True,
            "opaque_identifiers_are_internal_only": True,
        },
        "execution": {
            "this_prepare_module_has_execute_subcommand": False,
            "execution_allowed_by_this_prepare": False,
            "terminal_reason": (
                "this phase-A artifact stops after prepare; a separate live runner must "
                "freeze this exact artifact and receive separate review"
            ),
            "legacy_v1_v2_execution_paths_are_not_dependencies": True,
        },
        "scope": {
            "dynamic_caller_text_accepted_except_reserved_internal_id_tokens": True,
            "dynamic_reference_audio_paths_accepted": True,
            "natural_language_only_input_claimed": False,
            "semantic_style_match_claimed": False,
            "audio_quality_claimed": False,
            "future_channel_policy": (
                "mono target/reference audio is duplicated to stereo before the fixed "
                "stereo renderer; stereo input remains unchanged; converted slice hashes "
                "must be frozen by the future executor"
            ),
        },
        "reproducibility": {
            "code": _code_identities(),
            "source_manifest": dict(canonical.source_manifest),
        },
    }


def _staging_path(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise PrepareRunError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.staging_", dir=output.parent))


def prepare(
    *,
    manifest_path: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    base_url: str = llm.DEFAULT_BASE_URL,
) -> Path:
    output = output.resolve()
    staging = _staging_path(output)
    try:
        record = build_pre_registration(
            manifest_path=manifest_path,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
        )
        preregistration = staging / "PRE_REGISTRATION.json"
        _write_json_exclusive(preregistration, record)
        digest = manifest.file_sha256(preregistration)
        (staging / "PRE_REGISTRATION.sha256").write_text(
            f"{digest}  PRE_REGISTRATION.json\n", encoding="ascii"
        )
        _write_json_exclusive(
            staging / "PREPARATION_STATUS.json",
            {
                "status": "prepared_not_executed",
                "run_version": RUN_VERSION,
                "pre_registration_sha256": digest,
                "api_calls": 0,
                "audio_renders": 0,
                "optimizer_runs": 0,
                "optimizer_steps": 0,
                "audio_files_written": 0,
                "execution_allowed_by_this_prepare": False,
            },
        )
    except Exception as error:
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "status": "prepare_failed",
                "run_version": RUN_VERSION,
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


def verify_prepared(prepared: Path) -> tuple[dict[str, Any], str]:
    prepared = prepared.resolve()
    allowed = {
        "PRE_REGISTRATION.json", "PRE_REGISTRATION.sha256", "PREPARATION_STATUS.json"
    }
    if not prepared.is_dir() or {path.name for path in prepared.iterdir()} != allowed:
        raise PrepareRunError("prepared directory file set is invalid")
    preregistration = prepared / "PRE_REGISTRATION.json"
    sidecar = prepared / "PRE_REGISTRATION.sha256"
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = manifest.file_sha256(preregistration)
    if expected != actual:
        raise PrepareRunError("pre-registration hash changed")
    record = _read_json(preregistration)
    if record.get("run_version") != RUN_VERSION:
        raise PrepareRunError("prepared run version is not active v3")
    status = _read_json(prepared / "PREPARATION_STATUS.json")
    expected_status = {
        "status": "prepared_not_executed",
        "run_version": RUN_VERSION,
        "pre_registration_sha256": actual,
        "api_calls": 0,
        "audio_renders": 0,
        "optimizer_runs": 0,
        "optimizer_steps": 0,
        "audio_files_written": 0,
        "execution_allowed_by_this_prepare": False,
    }
    if status != expected_status:
        raise PrepareRunError("preparation status changed")
    for name, identity in record["reproducibility"]["code"].items():
        if _identity(Path(identity["path"])) != identity:
            raise PrepareRunError(f"frozen v3 code changed: {name}")
    recomputed = build_pre_registration(
        manifest_path=Path(record["reproducibility"]["source_manifest"]["path"]),
        model=record["endpoint_generation_config"]["model"],
        reasoning_effort=record["endpoint_generation_config"]["reasoning_effort"],
        base_url=record["endpoint_generation_config"]["base_url"],
    )
    if recomputed != record:
        raise PrepareRunError("prepared manifest, requests, or schema changed")
    return record, actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--model", default=DEFAULT_MODEL)
    prepare_parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    prepare_parser.add_argument("--base-url", default=llm.DEFAULT_BASE_URL)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        output = prepare(
            manifest_path=args.manifest,
            output=args.output,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            base_url=args.base_url,
        )
        result: Mapping[str, Any] = {"output": str(output)}
    else:
        _, digest = verify_prepared(args.prepared)
        result = {"prepared": str(args.prepared.resolve()), "sha256": digest}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
