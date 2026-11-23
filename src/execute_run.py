"""Runs a prepared request: endpoints, automation plan, curves and rendered audio."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf
import torch

import dsp_summary
import endpoint_interpolation as interpolation
import endpoint_manifest as manifest
import endpoint_runtime as masked_runtime
import llm
import planner
import prepare_run
import reference_match as reference
import reference_matcher as matcher
import renderer as renderer_module
import semantic_case as semantic_prereg
import text_endpoints as endpoint_contract
from automation_fx.reverb import DEFAULT_NUM_SAMPLES

LIVE_RUN_VERSION = "dynamic-multi-endpoint-v4-live-openai-api"
TEST_RUN_VERSION = "dynamic-multi-endpoint-v4-test-double"
LIVE_PLAN_SOURCE = "live_openai_api"
TEST_PLAN_SOURCE = "test_double"
SAMPLE_RATE = manifest.SAMPLE_RATE
BOUNDARY_SAMPLES = 441
REFERENCE_CLIP_FRAMES = 10 * SAMPLE_RATE
OFFICIAL_BASE_URL = llm.DEFAULT_BASE_URL
SEMANTIC_EVALUATOR_VERSION = semantic_prereg.EVALUATOR_VERSION
SEMANTIC_STATUS = semantic_prereg.STATUS
SEMANTIC_FILES = frozenset(
    {
        "SEMANTIC_PRE_REGISTRATION.json",
        "SEMANTIC_PRE_REGISTRATION.sha256",
        "PREPARATION_STATUS.json",
    }
)
AUTOMATION_SYSTEM_PROMPT = prepare_run.AUTOMATION_SYSTEM_PROMPT
_LIVE_OPENAI_CALL = llm.call_openai_structured_once
_LIVE_REFERENCE_MATCHER = matcher.run_reference_match
_LIVE_STATIC_VALIDATOR = endpoint_contract.validate_static_plan
_LIVE_STATIC_MATERIALIZER = endpoint_contract.normalized_parameters_from_plan
_LIVE_AUTOMATION_VALIDATOR = planner.validate_plan
_LIVE_REGISTRY_CLASS = masked_runtime.MaskedEndpointRegistry
_LIVE_FINAL_RENDERER_CLASS = masked_runtime.MaskedMultiEndpointRenderer
_LIVE_PARAMETER_EMBEDDER = masked_runtime.embed_endpoint_parameters
_LIVE_REFERENCE_RENDERER_CLASS = matcher.OfficialFxRenderer
_LIVE_PEAK_ENCODER_WRAPPER = reference.DifferentiablePeakNormalizedEncoder


def _callable_identity(function: Any) -> dict[str, Any]:
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    if not source_path.is_file():
        raise ExecuteRunError("runtime callable source is unavailable")
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source_file": _identity(source_path),
        "source_sha256": hashlib.sha256(
            inspect.getsource(function).encode("utf-8")
        ).hexdigest(),
    }


def _live_callable_identity() -> dict[str, Any]:
    function = _LIVE_OPENAI_CALL
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    expected_path = Path(llm.__file__).resolve()
    if (
        function is not llm.call_openai_structured_once
        or function.__module__ != "llm"
        or function.__name__ != "call_openai_structured_once"
        or source_path != expected_path
    ):
        raise ExecuteRunError(
            "live OpenAI transport callable identity changed"
        )
    return _callable_identity(function)


def _run_version(plan_source: str) -> str:
    if plan_source == LIVE_PLAN_SOURCE:
        return LIVE_RUN_VERSION
    if plan_source == TEST_PLAN_SOURCE:
        return TEST_RUN_VERSION
    raise ExecuteRunError(f"unsupported plan source {plan_source!r}")


def _transport_record(plan_source: str) -> dict[str, Any]:
    if plan_source == LIVE_PLAN_SOURCE:
        return {
            "kind": LIVE_PLAN_SOURCE,
            "module": "llm",
            "callable": "call_openai_structured_once",
            "module_identity": _identity(Path(llm.__file__)),
            "callable_identity": _live_callable_identity(),
            "test_double": False,
        }
    if plan_source == TEST_PLAN_SOURCE:
        return {
            "kind": TEST_PLAN_SOURCE,
            "module": None,
            "callable": "private_injected_test_callable",
            "module_identity": None,
            "callable_identity": None,
            "test_double": True,
        }
    raise ExecuteRunError(f"unsupported plan source {plan_source!r}")


class ExecuteRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RuntimeDependencies:
    source: str
    load_encoder: Callable[[Mapping[str, Any]], torch.nn.Module]
    match_reference: Callable[..., matcher.MatchResult]
    render_native_endpoint: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]
    validate_static_plan: Callable[..., dict[str, Any]]
    materialize_static_parameters: Callable[..., np.ndarray]
    validate_automation_plan: Callable[..., dict[str, Any]]
    embed_endpoint_parameters: Callable[..., torch.Tensor]
    reference_renderer_class: type[matcher.OfficialFxRenderer]
    registry_class: type[masked_runtime.MaskedEndpointRegistry]
    final_renderer_class: type[masked_runtime.MaskedMultiEndpointRenderer]


def _sha_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha_array(value: np.ndarray | torch.Tensor) -> str:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ExecuteRunError(f"missing source artifact: {path}")
    return {"path": str(path), "sha256": manifest.file_sha256(path)}


def _run_artifact(root: Path, path: Path) -> dict[str, str]:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ExecuteRunError(
            f"run artifact escaped its staging directory: {path}"
        ) from error
    if not path.is_file():
        raise ExecuteRunError(f"missing run artifact: {relative}")
    return {
        "relative_path": relative.as_posix(),
        "sha256": manifest.file_sha256(path),
    }


def _verify_run_artifact(root: Path, identity: Mapping[str, Any]) -> Path:
    if set(identity) != {"relative_path", "sha256"}:
        raise ExecuteRunError("run artifact identity fields changed")
    raw = identity["relative_path"]
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ExecuteRunError("run artifact path is not relative")
    root = root.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ExecuteRunError("run artifact path escaped run root") from error
    if not path.is_file() or manifest.file_sha256(path) != identity["sha256"]:
        raise ExecuteRunError(f"run artifact changed: {raw}")
    return path


def _artifact_manifest(root: Path) -> dict[str, str]:
    excluded = {"result.json", "COMPLETED.json", "FAILURE.json"}
    return {
        path.relative_to(root).as_posix(): manifest.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ExecuteRunError(f"expected JSON object: {path}")
    return value


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExecuteRunError(
            f"invalid JSON artifact {path}: {error}"
        ) from error


def _verify_semantic_prepared(
    semantic_prepared: Path,
    *,
    expected_prepared_path: Path,
    expected_prepared_digest: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    semantic_prepared = semantic_prepared.resolve()
    if (
        not semantic_prepared.is_dir()
        or {item.name for item in semantic_prepared.iterdir()} != SEMANTIC_FILES
    ):
        raise ExecuteRunError(
            "semantic preregistration file set is invalid"
        )
    prereg_path = semantic_prepared / "SEMANTIC_PRE_REGISTRATION.json"
    try:
        record, digest, status = semantic_prereg.verify_prepared_case(
            semantic_prepared
        )
    except Exception as error:
        raise ExecuteRunError(
            f"semantic preregistration verification failed: {error}"
        ) from error
    sidecar = semantic_prepared.joinpath(
        "SEMANTIC_PRE_REGISTRATION.sha256"
    ).read_text(encoding="ascii").split()
    if sidecar != [digest, "SEMANTIC_PRE_REGISTRATION.json"]:
        raise ExecuteRunError(
            "semantic preregistration sidecar changed"
        )
    source = record.get("source_binding")
    work = record.get("preparation_work")
    expected_work = {
        "api_calls": 0,
        "audio_renders": 0,
        "optimizer_runs": 0,
        "optimizer_steps": 0,
        "encoder_loads": 0,
        "audio_files_written": 0,
    }
    evaluator_code = record.get("evaluator_code")
    if (
        record.get("evaluator_version") != SEMANTIC_EVALUATOR_VERSION
        or record.get("status") != SEMANTIC_STATUS
        or record.get(
            "written_before_execution_pre_registration_and_any_api_call"
        )
        is not True
        or not isinstance(source, Mapping)
        or Path(str(source.get("v3_prepared_directory", ""))).resolve()
        != expected_prepared_path.resolve()
        or source.get("v3_pre_registration_sha256") != expected_prepared_digest
        or work != expected_work
        or not isinstance(evaluator_code, Mapping)
        or set(evaluator_code) != {"path", "sha256"}
        or _identity(Path(str(evaluator_code.get("path", ""))))
        != dict(evaluator_code)
    ):
        raise ExecuteRunError(
            "semantic preregistration contract or v3 binding changed"
        )
    sealed_at = status.get("sealed_at_unix_ns")
    expected_status = {
        "status": SEMANTIC_STATUS,
        "evaluator_version": SEMANTIC_EVALUATOR_VERSION,
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
        raise ExecuteRunError(
            "semantic preregistration status changed"
        )
    identity = {
        "directory": str(semantic_prepared),
        "pre_registration": _identity(prereg_path),
        "sidecar": _identity(
            semantic_prepared / "SEMANTIC_PRE_REGISTRATION.sha256"
        ),
        "preparation_status": _identity(
            semantic_prepared / "PREPARATION_STATUS.json"
        ),
        "verified_semantic_pre_registration_sha256": digest,
        "evaluator_version": SEMANTIC_EVALUATOR_VERSION,
    }
    return record, digest, identity


def _reverify_semantic_identity(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping) or set(identity) != {
        "directory",
        "pre_registration",
        "sidecar",
        "preparation_status",
        "verified_semantic_pre_registration_sha256",
        "evaluator_version",
    }:
        raise ExecuteRunError(
            "semantic preregistration identity fields changed"
        )
    directory = Path(str(identity["directory"])).resolve()
    if (
        identity.get("evaluator_version") != SEMANTIC_EVALUATOR_VERSION
        or not directory.is_dir()
        or {item.name for item in directory.iterdir()} != SEMANTIC_FILES
    ):
        raise ExecuteRunError(
            "semantic preregistration identity changed"
        )
    expected = {
        "pre_registration": _identity(
            directory / "SEMANTIC_PRE_REGISTRATION.json"
        ),
        "sidecar": _identity(
            directory / "SEMANTIC_PRE_REGISTRATION.sha256"
        ),
        "preparation_status": _identity(directory / "PREPARATION_STATUS.json"),
    }
    if any(identity[key] != value for key, value in expected.items()):
        raise ExecuteRunError(
            "semantic preregistration changed after execution preparation"
        )
    digest = expected["pre_registration"]["sha256"]
    if digest != identity["verified_semantic_pre_registration_sha256"]:
        raise ExecuteRunError(
            "semantic preregistration digest changed"
        )
    record = _read_json(directory / "SEMANTIC_PRE_REGISTRATION.json")
    evaluator_code = record.get("evaluator_code")
    if (
        not isinstance(evaluator_code, Mapping)
        or _identity(Path(str(evaluator_code.get("path", ""))))
        != dict(evaluator_code)
    ):
        raise ExecuteRunError(
            "semantic preregistration evaluator code changed"
        )
    return str(digest)


def _assert_semantic_preregistration_hidden(
    payload: Mapping[str, Any], semantic_identity: Mapping[str, Any]
) -> None:
    directory = Path(str(semantic_identity["directory"]))
    record = _read_json(directory / "SEMANTIC_PRE_REGISTRATION.json")
    forbidden_keys = {
        "case_nonce",
        "evidence",
        "scored_endpoint_ordinals",
        "expected_route",
        "qualitative_curve_constraints",
        "metric_config",
        "annotation_protocol",
        "selection_capability_claim_policy",
        "visibility_contract",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden_keys.intersection(str(key) for key in value)
            if overlap:
                raise ExecuteRunError(
                    "semantic evaluator fields leaked into an LLM payload"
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for secret in (
        record.get("case_nonce"),
        record.get("case_id"),
        record.get("evaluator_version"),
    ):
        if isinstance(secret, str) and secret and secret in serialized:
            raise ExecuteRunError(
                "semantic evaluator identity leaked into an LLM payload"
            )


def _reference_reproducibility() -> dict[str, Any]:
    result = reference._reproducibility_identity()
    result["installed_encoder_package"] = _fxencoder_package_identity()
    return result


def _fxencoder_package_identity() -> dict[str, Any]:
    spec = importlib.util.find_spec("fxencoder_plusplus")
    if spec is None or spec.origin is None:
        raise ExecuteRunError(
            "installed fxencoder_plusplus package cannot be located"
        )
    package_root = Path(spec.origin).resolve().parent
    files = {
        source.relative_to(package_root).as_posix(): manifest.file_sha256(source)
        for source in sorted(package_root.rglob("*.py"))
        if source.is_file()
    }
    if not files or "__init__.py" not in files or "model.py" not in files:
        raise ExecuteRunError(
            "installed fxencoder_plusplus Python source closure is incomplete"
        )
    try:
        version = importlib.metadata.version("fxencoder_plusplus")
    except importlib.metadata.PackageNotFoundError:
        version = importlib.metadata.version("fxencoder-plusplus")
    return {
        "package_root": str(package_root),
        "version": version,
        "python_files": files,
        "python_files_sha256": _sha_json(files),
    }


def _code_identities() -> dict[str, dict[str, str]]:
    modules = {
        "execution_runner": __import__(__name__),
        "semantic_preregistration": semantic_prereg,
        "masked_runtime": masked_runtime,
        "v3_prepare_runner": prepare_run,
        "manifest": manifest,
        "static_endpoint_contract": endpoint_contract,
        "planner": planner,
        "renderer": renderer_module,
        "endpoint_interpolation_renderer": interpolation,
        "dsp_summary": dsp_summary,
        "openai_schema_shim": llm,
        "openai_structured_call": llm,
        "matcher": matcher,
        "peak_normalized_match_runner": reference,
    }
    identities = {
        name: _identity(Path(module.__file__)) for name, module in modules.items()
    }
    identities["semantic_postrun_evaluator"] = _identity(
        Path(__file__).with_name(
            "evaluate_run.py"
        )
    )
    return identities


def _canonical_manifest(record: Mapping[str, Any]) -> manifest.CanonicalManifest:
    source = Path(record["reproducibility"]["source_manifest"]["path"])
    canonical = manifest.load_manifest(source)
    if canonical.to_dict() != record["canonical_manifest"]:
        raise ExecuteRunError(
            "current canonical manifest differs from finalized v3 preparation"
        )
    return canonical


def _timing(canonical: manifest.CanonicalManifest) -> dict[str, Any]:
    selected = canonical.target["selected_section_audio"]
    total = int(canonical.target["raw_stem"]["frames"])
    active_absolute_start = int(selected["start_sample"])
    active_absolute_stop = int(selected["stop_sample_exclusive"])
    desired_start = active_absolute_start - BOUNDARY_SAMPLES
    desired_stop = (
        active_absolute_stop + BOUNDARY_SAMPLES + DEFAULT_NUM_SAMPLES
    )
    source_start = max(0, desired_start)
    source_stop = min(total, desired_stop)
    pad_left = source_start - desired_start
    pad_right = desired_stop - source_stop
    active_start = pad_left + active_absolute_start - source_start
    active_stop = pad_left + active_absolute_stop - source_start
    release_stop = active_stop + BOUNDARY_SAMPLES
    render_samples = pad_left + source_stop - source_start + pad_right
    if not (
        0
        <= active_start - BOUNDARY_SAMPLES
        < active_start
        < active_stop
        < release_stop
        <= render_samples
        and release_stop + DEFAULT_NUM_SAMPLES <= render_samples
    ):
        raise ExecuteRunError("render timing lacks boundary or IR tail")
    return {
        "source_start_sample": source_start,
        "source_stop_sample_exclusive": source_stop,
        "left_zero_padding_samples": pad_left,
        "right_zero_padding_samples": pad_right,
        "render_samples": render_samples,
        "attack_start_sample_relative": active_start - BOUNDARY_SAMPLES,
        "active_start_sample_relative": active_start,
        "active_stop_sample_relative": active_stop,
        "release_stop_sample_relative": release_stop,
        "boundary_samples": BOUNDARY_SAMPLES,
        "ir_context_samples": DEFAULT_NUM_SAMPLES,
        "selected_source_start_sample": active_absolute_start,
        "selected_source_stop_sample_exclusive": active_absolute_stop,
        "edge_policy": (
            "real target context when present, otherwise deterministic zeros; full "
            "attack, release, and causal reverb-tail context retained"
        ),
    }


def _prepared_identity(prepared: Path, digest: str) -> dict[str, Any]:
    return {
        "directory": str(prepared.resolve()),
        "pre_registration": _identity(prepared / "PRE_REGISTRATION.json"),
        "sidecar": _identity(prepared / "PRE_REGISTRATION.sha256"),
        "preparation_status": _identity(prepared / "PREPARATION_STATUS.json"),
        "verified_pre_registration_sha256": digest,
    }


def _build_execution_pre_registration(
    *, prepared_path: Path, semantic_prepared: Path, plan_source: str
) -> dict[str, Any]:
    run_version = _run_version(plan_source)
    transport = _transport_record(plan_source)
    runtime_dependencies = _runtime_dependency_record(plan_source)
    prepared_record, prepared_digest = prepare_run.verify_prepared(prepared_path)
    _, semantic_digest, semantic_identity = _verify_semantic_prepared(
        semantic_prepared,
        expected_prepared_path=prepared_path,
        expected_prepared_digest=prepared_digest,
    )
    canonical = _canonical_manifest(prepared_record)
    config = prepared_record["endpoint_generation_config"]
    if config["base_url"].rstrip("/") != OFFICIAL_BASE_URL.rstrip("/"):
        raise ExecuteRunError(
            "live v3 currently permits only the official OpenAI API base URL"
        )
    text_count = sum(
        endpoint.kind == "natural_language" for endpoint in canonical.endpoints
    )
    reference_count = sum(
        endpoint.kind == "processed_reference" for endpoint in canonical.endpoints
    )
    for endpoint in canonical.endpoints:
        if endpoint.kind == "processed_reference" and (
            endpoint.source_audio is None
            or int(endpoint.source_audio["frames"]) != REFERENCE_CLIP_FRAMES
        ):
            raise ExecuteRunError(
                "each processed reference must be a caller-selected exact 10.0-second "
                "clip; full-song reference selection is outside this live-v3 scope"
            )
    reference_reproducibility = _reference_reproducibility()
    return {
        "run_version": run_version,
        "plan_source": plan_source,
        "transport": transport,
        "runtime_dependencies": runtime_dependencies,
        "plan_source_contract": {
            "persistent_artifacts_are_source_specific": True,
            "live_and_test_run_versions_are_distinct": True,
            "live_verifier_accepts_test_double": False,
            "test_verifier_accepts_live_openai_api": False,
            "production_cli_plan_source": LIVE_PLAN_SOURCE,
            "runtime_dependency_source_matches_plan_source": True,
        },
        "status": "execution_prepared_not_executed",
        "written_before_any_api_or_optimizer_call": True,
        "source_v3_prepared": _prepared_identity(prepared_path, prepared_digest),
        "semantic_evaluation_pre_registration": semantic_identity,
        "semantic_pre_registration_sha256": semantic_digest,
        "source_prepare_claim_scope": {
            "phase_a_artifact": True,
            "phase_a_execution_allowed_remains_false": True,
            "no_execute_subcommand_means_this_prepare_module_only": True,
            "separate_live_runner_identity_is_frozen_below": True,
        },
        "live_implementation_status": {
            "separate_reviewed_executor_present": True,
            "static_text_materializer_present": True,
            "processed_reference_materializer_present": True,
            "final_automation_request_materialized_only_during_reviewed_execution": True,
            "this_preregistration_is_the_reviewed_execution_input": True,
            "allows_execution_without_explicit_review": False,
            "review_gate": "execute requires reviewed=True after this artifact is frozen",
        },
        "canonical_manifest_sha256": _sha_json(canonical.to_dict()),
        "source_manifest": dict(canonical.source_manifest),
        "target": deepcopy(dict(canonical.target)),
        "timing": _timing(canonical),
        "endpoint_generation_config": deepcopy(dict(config)),
        "endpoint_materialization_routes_sha256": _sha_json(
            prepared_record["endpoint_materialization_routes"]
        ),
        "automation_schema": deepcopy(
            prepared_record["automation_planner"]["dynamic_schema"]
        ),
        "automation_schema_sha256": prepared_record["automation_planner"][
            "dynamic_schema_sha256"
        ],
        "automation_system_prompt": AUTOMATION_SYSTEM_PROMPT,
        "automation_system_prompt_sha256": hashlib.sha256(
            AUTOMATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "exact_work_budget": {
            "text_endpoint_api_calls": text_count,
            "automation_api_calls": 1,
            "total_api_calls": text_count + 1,
            "api_retries": 0,
            "json_repairs": 0,
            "reference_optimizer_runs": reference_count,
            "reference_optimizer_steps_per_run": matcher.MATCH_STEPS,
            "reference_optimizer_steps_total": reference_count
            * matcher.MATCH_STEPS,
            "encoder_loads": int(reference_count > 0),
            "endpoint_probe_audio_renders": text_count
            + reference_count * (matcher.MATCH_STEPS + 1),
            "final_automation_audio_renders": 1,
            "total_audio_renders": text_count
            + reference_count * (matcher.MATCH_STEPS + 1)
            + 1,
        },
        "actual_work_at_execution_prepare": {
            "api_calls": 0,
            "audio_renders": 0,
            "optimizer_runs": 0,
            "optimizer_steps": 0,
            "encoder_loads": 0,
            "audio_files_written": 0,
        },
        "endpoint_policy": {
            "natural_language": (
                "one exact caller-text static structured request; no timeline fields"
            ),
            "processed_reference": (
                "caller-selected exact 10.0-second processed reference clip only, "
                "matched against the current full target selected section with plain "
                "Fx-Encoder++ cosine; no reference dry"
            ),
            "processed_reference_frames": REFERENCE_CLIP_FRAMES,
            "processed_reference_full_song_selection": False,
            "target_match_extent": "current exact full selected section",
            "encoder_normalization": (
                "differentiable per-item absolute peak with true gradient"
            ),
            "native_match_then_embed": True,
            "embed_is_never_inside_optimizer_graph": True,
            "effect_applicability": (
                "immutable per-endpoint EQ/reverb mask, reverified before render"
            ),
        },
        "planner_visibility": {
            "exact_automation_instruction": True,
            "opaque_endpoint_ids": True,
            "exact_caller_endpoint_text": True,
            "caller_declared_effect_mode": True,
            "processed_reference_ordinal": True,
            "common_target_dsp_summaries": True,
            "music_structure": True,
            "audio_or_paths": False,
            "parameters": False,
            "embeddings": False,
            "optimizer_data": False,
            "provenance": False,
        },
        "render_policy": {
            "renderer": masked_runtime.RUNTIME_VERSION,
            "limiter": False,
            "trim_or_normalization": False,
            "raw_clipping_is_terminal": True,
            "incomplete_reverb_tail_is_terminal": True,
            "control_curves_saved_exactly": True,
            "forced_event_count_or_order_or_amount_or_final_endpoint": False,
        },
        "output_scope": {
            "target_raw_stem_section_with_boundary_and_tail_context": True,
            "full_mixture_remixed": False,
            "other_stems_loaded": False,
            "semantic_style_match_claimed": False,
        },
        "reproducibility": {
            "code": _code_identities(),
            "reference_runtime": reference_reproducibility,
        },
    }


def build_execution_pre_registration(
    *, prepared_path: Path, semantic_prepared: Path
) -> dict[str, Any]:
    return _build_execution_pre_registration(
        prepared_path=prepared_path,
        semantic_prepared=semantic_prepared,
        plan_source=LIVE_PLAN_SOURCE,
    )


def _build_test_execution_pre_registration(
    *, prepared_path: Path, semantic_prepared: Path
) -> dict[str, Any]:
    return _build_execution_pre_registration(
        prepared_path=prepared_path,
        semantic_prepared=semantic_prepared,
        plan_source=TEST_PLAN_SOURCE,
    )


def _staging_path(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise ExecuteRunError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.staging_", dir=output.parent))


def _prepare_execution(
    *, prepared_path: Path, semantic_prepared: Path, output: Path, plan_source: str
) -> Path:
    run_version = _run_version(plan_source)
    output = output.resolve()
    staging = _staging_path(output)
    try:
        record = _build_execution_pre_registration(
            prepared_path=prepared_path,
            semantic_prepared=semantic_prepared,
            plan_source=plan_source,
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
                "status": "execution_prepared_not_executed",
                "run_version": run_version,
                "plan_source": plan_source,
                "pre_registration_sha256": digest,
                **record["actual_work_at_execution_prepare"],
                "review_required": True,
            },
        )
    except Exception as error:
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "status": "execution_prepare_failed",
                "run_version": run_version,
                "plan_source": plan_source,
                "error_type": type(error).__name__,
                "error": str(error),
                "api_calls": 0,
                "optimizer_runs": 0,
                "traceback": traceback.format_exc(),
            },
        )
        os.replace(staging, output)
        raise
    os.replace(staging, output)
    return output


def prepare_execution(
    *, prepared_path: Path, semantic_prepared: Path, output: Path
) -> Path:
    return _prepare_execution(
        prepared_path=prepared_path,
        semantic_prepared=semantic_prepared,
        output=output,
        plan_source=LIVE_PLAN_SOURCE,
    )


def _prepare_test_execution(
    *, prepared_path: Path, semantic_prepared: Path, output: Path
) -> Path:
    return _prepare_execution(
        prepared_path=prepared_path,
        semantic_prepared=semantic_prepared,
        output=output,
        plan_source=TEST_PLAN_SOURCE,
    )


def _verify_execution_prepared(
    prepared: Path, *, expected_plan_source: str
) -> tuple[dict[str, Any], str]:
    expected_run_version = _run_version(expected_plan_source)
    prepared = prepared.resolve()
    allowed = {
        "PRE_REGISTRATION.json",
        "PRE_REGISTRATION.sha256",
        "PREPARATION_STATUS.json",
    }
    if not prepared.is_dir() or {path.name for path in prepared.iterdir()} != allowed:
        raise ExecuteRunError("execution prepared file set is invalid")
    preregistration = prepared / "PRE_REGISTRATION.json"
    digest = manifest.file_sha256(preregistration)
    if prepared.joinpath("PRE_REGISTRATION.sha256").read_text(
        encoding="ascii"
    ).split()[0] != digest:
        raise ExecuteRunError("execution preregistration hash changed")
    record = _read_json(preregistration)
    if (
        record.get("run_version") != expected_run_version
        or record.get("plan_source") != expected_plan_source
        or record.get("transport") != _transport_record(expected_plan_source)
    ):
        raise ExecuteRunError("execution preregistration version changed")
    status = _read_json(prepared / "PREPARATION_STATUS.json")
    expected_status = {
        "status": "execution_prepared_not_executed",
        "run_version": expected_run_version,
        "plan_source": expected_plan_source,
        "pre_registration_sha256": digest,
        **record["actual_work_at_execution_prepare"],
        "review_required": True,
    }
    if status != expected_status:
        raise ExecuteRunError("execution preparation status changed")
    for name, identity in record["reproducibility"]["code"].items():
        if _identity(Path(identity["path"])) != identity:
            raise ExecuteRunError(f"frozen live code changed: {name}")
    reference_runtime = record["reproducibility"]["reference_runtime"]
    checkpoint = reference_runtime["checkpoint"]
    if _identity(Path(checkpoint["path"])) != {
        "path": checkpoint["path"],
        "sha256": checkpoint["sha256"],
    }:
        raise ExecuteRunError("Fx-Encoder++ checkpoint changed")
    recomputed = _build_execution_pre_registration(
        prepared_path=Path(record["source_v3_prepared"]["directory"]),
        semantic_prepared=Path(
            record["semantic_evaluation_pre_registration"]["directory"]
        ),
        plan_source=expected_plan_source,
    )
    if recomputed != record:
        raise ExecuteRunError(
            "execution sources, requests, code, or policy changed"
        )
    return record, digest


def verify_execution_prepared(prepared: Path) -> tuple[dict[str, Any], str]:
    return _verify_execution_prepared(
        prepared, expected_plan_source=LIVE_PLAN_SOURCE
    )


def _verify_test_execution_prepared(
    prepared: Path,
) -> tuple[dict[str, Any], str]:
    return _verify_execution_prepared(
        prepared, expected_plan_source=TEST_PLAN_SOURCE
    )


def _load_audio(identity: Mapping[str, Any]) -> np.ndarray:
    path = Path(identity["path"])
    if _identity(path) != {"path": str(path.resolve()), "sha256": identity["sha256"]}:
        raise ExecuteRunError(f"audio source hash changed: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] not in (1, 2):
        raise ExecuteRunError("audio source format changed")
    if "frames" in identity and int(identity["frames"]) != int(audio.shape[0]):
        raise ExecuteRunError("audio source frame count changed")
    if "channels" in identity and int(identity["channels"]) != int(audio.shape[1]):
        raise ExecuteRunError("audio source channel count changed")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not bool(np.isfinite(audio).all()):
        raise ExecuteRunError("decoded audio is not finite")
    if _identity(path)["sha256"] != identity["sha256"]:
        raise ExecuteRunError("audio source changed while decoded")
    return audio


def _load_reference_clip(identity: Mapping[str, Any]) -> np.ndarray:
    if int(identity.get("frames", -1)) != REFERENCE_CLIP_FRAMES:
        raise ExecuteRunError(
            "processed reference is not the frozen exact 10.0-second clip"
        )
    audio = _load_audio(identity)
    if audio.shape != (REFERENCE_CLIP_FRAMES, 2):
        raise ExecuteRunError(
            "decoded processed reference length or stereo policy changed"
        )
    return audio


def _verify_source_identities(canonical: manifest.CanonicalManifest) -> None:
    sources: list[Mapping[str, Any]] = [
        canonical.source_manifest,
        canonical.target["raw_stem"],
        canonical.target["structure"],
    ]
    sources.extend(
        endpoint.source_audio
        for endpoint in canonical.endpoints
        if endpoint.source_audio is not None
    )
    for source in sources:
        path = Path(source["path"])
        if _identity(path) != {
            "path": str(path.resolve()),
            "sha256": source["sha256"],
        }:
            raise ExecuteRunError(f"source identity changed: {path}")


def _tensor_audio(audio: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(audio.T[None], dtype=torch.float32)


def _safe_audio(audio: np.ndarray, where: str) -> dict[str, Any]:
    peak = float(np.max(np.abs(audio), initial=0.0))
    clipping = int(np.count_nonzero(np.abs(audio) > 1.0))
    if not bool(np.isfinite(audio).all()) or clipping:
        raise ExecuteRunError(
            f"{where} raw safety failed: peak={peak}, clipping_samples={clipping}"
        )
    return {
        "finite": True,
        "peak_abs": peak,
        "samples_over_1_0": clipping,
        "limiter": False,
        "trim_or_normalization": False,
    }


def _load_encoder(reference_runtime: Mapping[str, Any]) -> torch.nn.Module:
    if reference_runtime.get("installed_encoder_package") != _fxencoder_package_identity():
        raise ExecuteRunError(
            "installed fxencoder_plusplus package changed after preregistration"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    checkpoint = Path(reference_runtime["checkpoint"]["path"])
    from fxencoder_plusplus import load_model

    base = load_model(
        "default", model_path=str(checkpoint), device="cpu", auto_download=False
    )
    return _LIVE_PEAK_ENCODER_WRAPPER(base)


def _render_native_endpoint(
    target: torch.Tensor, parameters: torch.Tensor, effect_mode: str
) -> torch.Tensor:
    renderer = _LIVE_REFERENCE_RENDERER_CLASS(
        sample_rate=SAMPLE_RATE,
        device="cpu",
        reverb_seed=matcher.DEFAULT_REVERB_SEED,
    )
    with torch.no_grad():
        return renderer.render(target, parameters, effect_mode).detach().clone()


_LIVE_ENCODER_LOADER = _load_encoder
_LIVE_NATIVE_ENDPOINT_RENDERER = _render_native_endpoint


def _assert_live_runtime_callables() -> None:
    bindings = (
        (_load_encoder, _LIVE_ENCODER_LOADER, "encoder loader"),
        (
            matcher.run_reference_match,
            _LIVE_REFERENCE_MATCHER,
            "reference matcher",
        ),
        (
            _render_native_endpoint,
            _LIVE_NATIVE_ENDPOINT_RENDERER,
            "native endpoint renderer",
        ),
        (
            endpoint_contract.validate_static_plan,
            _LIVE_STATIC_VALIDATOR,
            "static endpoint validator",
        ),
        (
            endpoint_contract.normalized_parameters_from_plan,
            _LIVE_STATIC_MATERIALIZER,
            "static endpoint materializer",
        ),
        (planner.validate_plan, _LIVE_AUTOMATION_VALIDATOR, "automation validator"),
        (
            masked_runtime.MaskedEndpointRegistry,
            _LIVE_REGISTRY_CLASS,
            "endpoint registry",
        ),
        (
            masked_runtime.MaskedMultiEndpointRenderer,
            _LIVE_FINAL_RENDERER_CLASS,
            "final renderer",
        ),
        (
            masked_runtime.embed_endpoint_parameters,
            _LIVE_PARAMETER_EMBEDDER,
            "endpoint parameter embedder",
        ),
        (
            matcher.OfficialFxRenderer,
            _LIVE_REFERENCE_RENDERER_CLASS,
            "reference renderer",
        ),
        (
            reference.DifferentiablePeakNormalizedEncoder,
            _LIVE_PEAK_ENCODER_WRAPPER,
            "peak-normalized encoder wrapper",
        ),
    )
    changed = [name for current, frozen, name in bindings if current is not frozen]
    if changed:
        raise ExecuteRunError(
            f"live runtime callable identity changed: {', '.join(changed)}"
        )


def _live_runtime_dependencies() -> _RuntimeDependencies:
    _assert_live_runtime_callables()
    return _RuntimeDependencies(
        source=LIVE_PLAN_SOURCE,
        load_encoder=_LIVE_ENCODER_LOADER,
        match_reference=_LIVE_REFERENCE_MATCHER,
        render_native_endpoint=_LIVE_NATIVE_ENDPOINT_RENDERER,
        validate_static_plan=_LIVE_STATIC_VALIDATOR,
        materialize_static_parameters=_LIVE_STATIC_MATERIALIZER,
        validate_automation_plan=_LIVE_AUTOMATION_VALIDATOR,
        embed_endpoint_parameters=_LIVE_PARAMETER_EMBEDDER,
        reference_renderer_class=_LIVE_REFERENCE_RENDERER_CLASS,
        registry_class=_LIVE_REGISTRY_CLASS,
        final_renderer_class=_LIVE_FINAL_RENDERER_CLASS,
    )


def _make_test_runtime_dependencies(
    *,
    load_encoder: Callable[[Mapping[str, Any]], torch.nn.Module],
    match_reference: Callable[..., matcher.MatchResult],
    render_native_endpoint: Callable[
        [torch.Tensor, torch.Tensor, str], torch.Tensor
    ] = _LIVE_NATIVE_ENDPOINT_RENDERER,
) -> _RuntimeDependencies:
    return _RuntimeDependencies(
        source=TEST_PLAN_SOURCE,
        load_encoder=load_encoder,
        match_reference=match_reference,
        render_native_endpoint=render_native_endpoint,
        validate_static_plan=_LIVE_STATIC_VALIDATOR,
        materialize_static_parameters=_LIVE_STATIC_MATERIALIZER,
        validate_automation_plan=_LIVE_AUTOMATION_VALIDATOR,
        embed_endpoint_parameters=_LIVE_PARAMETER_EMBEDDER,
        reference_renderer_class=_LIVE_REFERENCE_RENDERER_CLASS,
        registry_class=_LIVE_REGISTRY_CLASS,
        final_renderer_class=_LIVE_FINAL_RENDERER_CLASS,
    )


def _runtime_dependency_record(plan_source: str) -> dict[str, Any]:
    if plan_source == TEST_PLAN_SOURCE:
        return {
            "source": TEST_PLAN_SOURCE,
            "explicit_private_dependency_injection": True,
            "live_callable_identities": None,
        }
    dependencies = _live_runtime_dependencies()
    callables = {
        "load_encoder": dependencies.load_encoder,
        "match_reference": dependencies.match_reference,
        "render_native_endpoint": dependencies.render_native_endpoint,
        "validate_static_plan": dependencies.validate_static_plan,
        "materialize_static_parameters": dependencies.materialize_static_parameters,
        "validate_automation_plan": dependencies.validate_automation_plan,
        "embed_endpoint_parameters": dependencies.embed_endpoint_parameters,
        "reference_renderer_class": dependencies.reference_renderer_class,
        "registry_class": dependencies.registry_class,
        "final_renderer_class": dependencies.final_renderer_class,
    }
    return {
        "source": LIVE_PLAN_SOURCE,
        "explicit_private_dependency_injection": False,
        "live_callable_identities": {
            name: _callable_identity(function)
            for name, function in callables.items()
        },
    }


def _validate_runtime_dependency_bundle(
    plan_source: str, dependencies: _RuntimeDependencies
) -> dict[str, Any]:
    if not isinstance(dependencies, _RuntimeDependencies):
        raise ExecuteRunError(
            "runtime dependencies must use the private immutable bundle"
        )
    if dependencies.source != plan_source:
        raise ExecuteRunError(
            "runtime dependency source and plan source disagree"
        )
    record = _runtime_dependency_record(plan_source)
    if plan_source == LIVE_PLAN_SOURCE:
        frozen = _live_runtime_dependencies()
        for field in (
            "load_encoder",
            "match_reference",
            "render_native_endpoint",
            "validate_static_plan",
            "materialize_static_parameters",
            "validate_automation_plan",
            "embed_endpoint_parameters",
            "reference_renderer_class",
            "registry_class",
            "final_renderer_class",
        ):
            if getattr(dependencies, field) is not getattr(frozen, field):
                raise ExecuteRunError(
                    f"live runtime dependency changed: {field}"
                )
    return record


def _native_request(payload: Mapping[str, Any], *, base_url: str, reasoning_effort: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": payload["model"],
        "messages": payload["messages"],
        "response_format": llm._strict_response_format(
            payload["response_format"]
        ),
    }
    if llm.is_reasoning_model(str(payload["model"])):
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
    else:
        body["temperature"] = float(payload.get("temperature", 0.0))
    return {
        "endpoint": f"{base_url.rstrip('/')}/chat/completions",
        "body": body,
        "api_key_recorded": False,
    }


def _call_once(
    *,
    payload: dict[str, Any],
    base_url: str,
    reasoning_effort: str | None,
    timeout: float,
    output_directory: Path,
    plan_source: str,
    runtime_dependencies: Mapping[str, Any],
    semantic_pre_registration: Mapping[str, Any],
    call_once: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    run_version = _run_version(plan_source)
    transport = _transport_record(plan_source)
    semantic_digest = _reverify_semantic_identity(semantic_pre_registration)
    _assert_semantic_preregistration_hidden(payload, semantic_pre_registration)
    expected_native = _native_request(
        payload, base_url=base_url, reasoning_effort=reasoning_effort
    )
    _write_json_exclusive(
        output_directory / "request.json",
        {
            "run_version": run_version,
            "plan_source": plan_source,
            "transport": transport,
            "runtime_dependencies": dict(runtime_dependencies),
            "semantic_pre_registration_sha256": semantic_digest,
            "payload": payload,
            "native_request": expected_native,
            "retry": 0,
            "repair": 0,
        },
    )
    if _reverify_semantic_identity(semantic_pre_registration) != semantic_digest:
        raise ExecuteRunError(
            "semantic preregistration changed immediately before API call"
        )
    response, native = call_once(
        payload,
        base_url=base_url,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )
    _write_json_exclusive(
        output_directory / "raw_response_before_parse.json",
        {
            "run_version": run_version,
            "plan_source": plan_source,
            "transport": transport,
            "runtime_dependencies": dict(runtime_dependencies),
            "semantic_pre_registration_sha256": semantic_digest,
            "response": response,
            "native_request": native,
        },
    )
    if native != expected_native:
        raise ExecuteRunError("actual native API request changed")
    meta = response.get("native_meta")
    if not isinstance(meta, Mapping) or meta.get("finish_reason") != "stop":
        raise ExecuteRunError("structured call did not finish with stop")
    try:
        content = response["choices"][0]["message"]["content"]
        raw = json.loads(content)
    except Exception as error:
        raise ExecuteRunError(
            f"structured response JSON parse failed: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ExecuteRunError("structured response is not an object")
    _write_json_exclusive(output_directory / "raw_plan.json", raw)
    return raw


def _verify_saved_call(
    *,
    run_root: Path,
    call_directory: Path,
    expected_payload: Mapping[str, Any],
    base_url: str,
    reasoning_effort: str | None,
    expected_plan_source: str,
    expected_runtime_dependencies: Mapping[str, Any],
    expected_semantic_pre_registration: Mapping[str, Any],
) -> dict[str, Any]:
    expected_run_version = _run_version(expected_plan_source)
    transport = _transport_record(expected_plan_source)
    semantic_digest = _reverify_semantic_identity(
        expected_semantic_pre_registration
    )
    _assert_semantic_preregistration_hidden(
        expected_payload, expected_semantic_pre_registration
    )
    request = _read_json(call_directory / "request.json")
    expected_native = _native_request(
        expected_payload,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )
    if request != {
        "run_version": expected_run_version,
        "plan_source": expected_plan_source,
        "transport": transport,
        "runtime_dependencies": dict(expected_runtime_dependencies),
        "semantic_pre_registration_sha256": semantic_digest,
        "payload": expected_payload,
        "native_request": expected_native,
        "retry": 0,
        "repair": 0,
    }:
        raise ExecuteRunError("saved structured request changed")
    saved_response = _read_json(
        call_directory / "raw_response_before_parse.json"
    )
    if (
        set(saved_response)
        != {
            "run_version",
            "plan_source",
            "transport",
            "runtime_dependencies",
            "semantic_pre_registration_sha256",
            "response",
            "native_request",
        }
        or saved_response["run_version"] != expected_run_version
        or saved_response["plan_source"] != expected_plan_source
        or saved_response["transport"] != transport
        or saved_response["runtime_dependencies"]
        != dict(expected_runtime_dependencies)
        or saved_response["semantic_pre_registration_sha256"]
        != semantic_digest
        or saved_response["native_request"] != expected_native
    ):
        raise ExecuteRunError(
            "saved native request and response provenance disagree"
        )
    response = saved_response["response"]
    if not isinstance(response, Mapping):
        raise ExecuteRunError("saved API response is not an object")
    meta = response.get("native_meta")
    if not isinstance(meta, Mapping) or meta.get("finish_reason") != "stop":
        raise ExecuteRunError("saved API finish reason changed")
    try:
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as error:
        raise ExecuteRunError(
            f"saved API response cannot be parsed: {error}"
        ) from error
    if not isinstance(parsed, dict) or _read_json(
        call_directory / "raw_plan.json"
    ) != parsed:
        raise ExecuteRunError(
            "saved raw response and raw plan disagree"
        )
    for artifact in (
        call_directory / "request.json",
        call_directory / "raw_response_before_parse.json",
        call_directory / "raw_plan.json",
    ):
        _verify_run_artifact(run_root, _run_artifact(run_root, artifact))
    return parsed


def _structure_context(canonical: manifest.CanonicalManifest) -> dict[str, Any]:
    structure_identity = canonical.target["structure"]
    structure_path = Path(structure_identity["path"])
    expected_identity = {
        "path": str(structure_path.resolve()),
        "sha256": structure_identity["sha256"],
    }
    if _identity(structure_path) != expected_identity:
        raise ExecuteRunError("music-structure source hash changed")
    raw = _read_json(structure_path)
    selected = canonical.target["structure"]["selected_section"]
    start = float(selected["start_sec"])
    stop = float(selected["end_sec"])
    actual = [
        float(value)
        for value in raw.get("downbeats", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and start <= float(value) <= stop
    ]
    boundaries = sorted(set([start, stop] + [value for value in actual if start < value < stop]))
    grid = boundaries
    bars = [
        {"bar_index": index, "start_sec": left, "end_sec": right}
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]), start=1)
    ]
    if not bars:
        raise ExecuteRunError("selected structure has no bar interval")
    beats = [
        float(value)
        for value in raw.get("beats", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and start <= float(value) <= stop
    ]
    bpm = raw.get("bpm")
    if (
        isinstance(bpm, bool)
        or not isinstance(bpm, (int, float))
        or not np.isfinite(float(bpm))
        or float(bpm) <= 0.0
    ):
        raise ExecuteRunError(
            "trusted music-structure source lacks a positive finite BPM"
        )
    result = {
        "section_scope": "selected_only",
        "bpm": float(bpm),
        "n_beats": max(len(beats), len(grid)),
        "n_downbeats": len(grid),
        "downbeats_seconds": grid,
        "available_sections": [
            {
                "label": canonical.target["section"],
                "occurrence": int(canonical.target["occurrence"]),
                "start_sec": start,
                "end_sec": stop,
            }
        ],
        "section_bar_boundaries": [
            {
                "label": canonical.target["section"],
                "occurrence": int(canonical.target["occurrence"]),
                "section_start_sec": start,
                "section_end_sec": stop,
                "bar_count": len(bars),
                "bars": bars,
            }
        ],
    }
    if _identity(structure_path) != expected_identity:
        raise ExecuteRunError(
            "music-structure source changed while context was built"
        )
    try:
        validated = planner._validate_structure_summary(result)
    except planner.MultiEndpointPlanError as error:
        raise ExecuteRunError(str(error)) from error
    if validated != result:
        raise ExecuteRunError(
            "music-structure context canonicalization changed"
        )
    return result


def _automation_context(
    *,
    canonical: manifest.CanonicalManifest,
    structure_context: Mapping[str, Any],
    endpoint_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    reference_ordinal = 0
    for record in endpoint_records:
        role = record["role"]
        view: dict[str, Any] = {
            "endpoint_id": record["endpoint_id"],
            "role": role,
            "caller_declared_effect_mode": record["endpoint_mode"],
            "common_target_dsp_summary": record["common_target_probe"]["dsp_summary"],
        }
        if role == "caller_natural_language":
            view["caller_description_exact"] = record["caller_description_exact"]
        elif role == "processed_reference":
            reference_ordinal += 1
            view["processed_reference_ordinal"] = reference_ordinal
        views.append(view)
    context = {
        "INSTRUCTION": canonical.automation_instruction,
        "AVAILABLE_MUSIC_STRUCTURE": deepcopy(dict(structure_context)),
        "LOCKED_EFFECT_MODE": canonical.renderer_mode,
        "TARGET_DSP_SUMMARY": endpoint_records[0]["common_target_probe"]["dsp_summary"],
        "ENDPOINTS": views,
        "VISIBILITY_CONTRACT": {
            "exact_caller_text": True,
            "common_target_dsp_summaries": True,
            "raw_audio": False,
            "audio_paths": False,
            "embeddings": False,
            "dsp_parameters": False,
            "optimizer_data": False,
            "endpoint_provenance": False,
        },
    }
    _audit_automation_context(context, canonical=canonical, endpoint_records=endpoint_records)
    return context


def _audit_automation_context(
    context: Mapping[str, Any],
    *,
    canonical: manifest.CanonicalManifest,
    endpoint_records: list[Mapping[str, Any]],
) -> None:
    if set(context) != {
        "INSTRUCTION",
        "AVAILABLE_MUSIC_STRUCTURE",
        "LOCKED_EFFECT_MODE",
        "TARGET_DSP_SUMMARY",
        "ENDPOINTS",
        "VISIBILITY_CONTRACT",
    }:
        raise ExecuteRunError("automation context fields changed")
    if context.get("INSTRUCTION") != canonical.automation_instruction:
        raise ExecuteRunError("automation instruction was rewritten")
    views = context.get("ENDPOINTS")
    if not isinstance(views, list) or [row.get("endpoint_id") for row in views] != [
        row["endpoint_id"] for row in endpoint_records
    ]:
        raise ExecuteRunError("automation endpoint order changed")
    if context.get("LOCKED_EFFECT_MODE") != canonical.renderer_mode:
        raise ExecuteRunError("automation renderer mode changed")
    if context.get("TARGET_DSP_SUMMARY") != endpoint_records[0][
        "common_target_probe"
    ]["dsp_summary"]:
        raise ExecuteRunError("automation target summary changed")
    if context.get("VISIBILITY_CONTRACT") != {
        "exact_caller_text": True,
        "common_target_dsp_summaries": True,
        "raw_audio": False,
        "audio_paths": False,
        "embeddings": False,
        "dsp_parameters": False,
        "optimizer_data": False,
        "endpoint_provenance": False,
    }:
        raise ExecuteRunError("automation visibility contract changed")
    try:
        if planner._validate_structure_summary(
            context["AVAILABLE_MUSIC_STRUCTURE"]
        ) != context["AVAILABLE_MUSIC_STRUCTURE"]:
            raise ExecuteRunError(
                "automation structure context canonicalization changed"
            )
        for index, row in enumerate(views):
            if planner._validate_dsp_summary(
                row.get("common_target_dsp_summary"), f"ENDPOINTS[{index}] summary"
            ) != row.get("common_target_dsp_summary"):
                raise ExecuteRunError(
                    "automation endpoint summary canonicalization changed"
                )
    except planner.MultiEndpointPlanError as error:
        raise ExecuteRunError(str(error)) from error
    expected_text = {
        endpoint.endpoint_id: endpoint.user_description
        for endpoint in canonical.endpoints
        if endpoint.kind == "natural_language"
    }
    for row in views:
        expected_fields = {
            "endpoint_id",
            "role",
            "caller_declared_effect_mode",
            "common_target_dsp_summary",
        }
        if row["role"] == "caller_natural_language":
            expected_fields.add("caller_description_exact")
        elif row["role"] == "processed_reference":
            expected_fields.add("processed_reference_ordinal")
        elif row["role"] != "dry":
            raise ExecuteRunError("automation endpoint role changed")
        if set(row) != expected_fields:
            raise ExecuteRunError(
                "automation endpoint view fields changed"
            )
        matching = next(
            record
            for record in endpoint_records
            if record["endpoint_id"] == row["endpoint_id"]
        )
        if row["caller_declared_effect_mode"] != matching["endpoint_mode"]:
            raise ExecuteRunError("endpoint effect mode changed")
        if row["role"] == "caller_natural_language" and row.get(
            "caller_description_exact"
        ) != expected_text[row["endpoint_id"]]:
            raise ExecuteRunError("caller endpoint text changed")


def _automation_payload(
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    config = record["endpoint_generation_config"]
    return {
        "model": config["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": record["automation_system_prompt"]},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "dynamic_multi_endpoint_automation_v3",
                "strict": True,
                "schema": record["automation_schema"],
            },
        },
    }


def _endpoint_record(
    *,
    canonical_endpoint: manifest.CanonicalEndpoint | None,
    endpoint_id: str,
    role: str,
    endpoint_mode: str | None,
    native_parameters: torch.Tensor | None,
    embedded_parameters: torch.Tensor | None,
    common_target_audio: torch.Tensor,
    source: Mapping[str, Any],
    caller_description_exact: str | None = None,
) -> tuple[dict[str, Any], masked_runtime.MaskedEndpointSpec]:
    common_np = np.ascontiguousarray(common_target_audio[0].T.detach().cpu().numpy())
    safety = _safe_audio(common_np, f"{endpoint_id} common-target endpoint")
    summary = dsp_summary.summarize_audio(common_np, SAMPLE_RATE)
    summary_hash = _sha_json(summary)
    attestation = masked_runtime.applicability_attestation(
        endpoint_id=endpoint_id,
        endpoint_mode=endpoint_mode,
        renderer_mode=str(source["renderer_mode"]),
    )
    if embedded_parameters is None:
        provenance = {
            "verification_method": "exact target selected-slice identity",
            **attestation,
        }
    else:
        provenance = {
            "verification_status": "verified",
            "verification_method": str(source["verification_method"]),
            "source_artifacts_verified": True,
            "normalized_parameters_sha256": _sha_array(embedded_parameters),
            "dsp_summary_sha256": summary_hash,
            **attestation,
        }
    record = {
        "endpoint_id": endpoint_id,
        "identifier_visibility": "hidden_internal_identifier",
        "role": role,
        "endpoint_mode": endpoint_mode,
        "effect_applicability": attestation,
        "native_normalized_parameters": (
            None if native_parameters is None else native_parameters.tolist()
        ),
        "embedded_normalized_parameters": (
            None if embedded_parameters is None else embedded_parameters.tolist()
        ),
        "source": deepcopy(dict(source)),
        "common_target_probe": {
            "audio_array_sha256": _sha_array(common_np),
            "dsp_summary": summary,
            "dsp_summary_sha256": summary_hash,
            "safety": safety,
        },
    }
    if caller_description_exact is not None:
        record["caller_description_exact"] = caller_description_exact
        record["caller_description_utf8_sha256"] = hashlib.sha256(
            caller_description_exact.encode("utf-8")
        ).hexdigest()
    spec = masked_runtime.MaskedEndpointSpec(
        endpoint_id=endpoint_id,
        role=role,
        endpoint_mode=endpoint_mode,
        normalized_parameters=native_parameters,
        dsp_summary=summary,
        provenance=provenance,
    )
    return record, spec


def _render_source(target: np.ndarray, timing: Mapping[str, Any]) -> np.ndarray:
    source = target[
        int(timing["source_start_sample"]) : int(timing["source_stop_sample_exclusive"])
    ]
    left = np.zeros((int(timing["left_zero_padding_samples"]), 2), dtype=np.float32)
    right = np.zeros((int(timing["right_zero_padding_samples"]), 2), dtype=np.float32)
    result = np.ascontiguousarray(np.concatenate((left, source, right), axis=0))
    if result.shape != (int(timing["render_samples"]), 2):
        raise ExecuteRunError("render source length changed")
    return result


def _write_float(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")


def _execute(
    *,
    prepared: Path,
    output: Path,
    reviewed: bool,
    plan_source: str,
    call_once: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
    runtime_dependencies: _RuntimeDependencies,
    timeout: float = 900.0,
) -> Path:
    run_version = _run_version(plan_source)
    transport = _transport_record(plan_source)
    runtime_dependency_record = _validate_runtime_dependency_bundle(
        plan_source, runtime_dependencies
    )
    if plan_source == LIVE_PLAN_SOURCE and call_once is not _LIVE_OPENAI_CALL:
        raise ExecuteRunError("live execution transport changed")
    if plan_source == TEST_PLAN_SOURCE and call_once is _LIVE_OPENAI_CALL:
        raise ExecuteRunError(
            "test execution requires an explicit non-live test double"
        )
    if not reviewed:
        raise ExecuteRunError("live execution requires explicit review")
    output = output.resolve()
    staging = _staging_path(output)
    started = time.perf_counter()
    counters = {
        "api_calls_started": 0,
        "api_calls_completed": 0,
        "optimizer_runs_started": 0,
        "optimizer_runs_completed": 0,
        "optimizer_steps_completed": 0,
        "encoder_loads_completed": 0,
        "audio_renders_completed": 0,
    }
    try:
        record, prereg_digest = _verify_execution_prepared(
            prepared, expected_plan_source=plan_source
        )
        if record.get("runtime_dependencies") != runtime_dependency_record:
            raise ExecuteRunError(
                "prepared runtime dependency source changed"
            )
        semantic_identity = record["semantic_evaluation_pre_registration"]
        semantic_digest = _reverify_semantic_identity(semantic_identity)
        if semantic_digest != record["semantic_pre_registration_sha256"]:
            raise ExecuteRunError(
                "prepared semantic preregistration digest changed"
            )
        shutil.copy2(prepared / "PRE_REGISTRATION.json", staging / "PRE_REGISTRATION.json")
        (staging / "PRE_REGISTRATION.sha256").write_text(
            f"{prereg_digest}  PRE_REGISTRATION.json\n", encoding="ascii"
        )
        shutil.copytree(
            Path(semantic_identity["directory"]),
            staging / "semantic_pre_registration",
        )
        prepared_record, _ = prepare_run.verify_prepared(
            Path(record["source_v3_prepared"]["directory"])
        )
        canonical = _canonical_manifest(prepared_record)
        _verify_source_identities(canonical)
        target = _load_audio(canonical.target["raw_stem"])
        selected = target[
            int(record["timing"]["selected_source_start_sample"]) :
            int(record["timing"]["selected_source_stop_sample_exclusive"])
        ]
        selected_hash = _sha_array(selected)
        if selected_hash != canonical.target["selected_section_audio"]["decoded_stereo_sha256"]:
            raise ExecuteRunError("current target selected slice hash changed")
        selected_tensor = _tensor_audio(selected)
        _write_json_exclusive(
            staging / "source_snapshot.json",
            {
                "manifest": dict(canonical.source_manifest),
                "target_file": dict(canonical.target["raw_stem"]),
                "structure_file": {
                    key: canonical.target["structure"][key]
                    for key in ("path", "sha256")
                },
                "target_decoded_stereo_sha256": _sha_array(target),
                "target_selected_stereo_sha256": selected_hash,
                "reference_files": [
                    dict(endpoint.source_audio)
                    for endpoint in canonical.endpoints
                    if endpoint.source_audio is not None
                ],
                "processed_reference_policy": {
                    "caller_selected_precropped_clip": True,
                    "required_frames": REFERENCE_CLIP_FRAMES,
                    "required_duration_sec": 10.0,
                    "full_song_selection_performed": False,
                },
            },
        )

        endpoint_records: list[dict[str, Any]] = []
        endpoint_specs: list[masked_runtime.MaskedEndpointSpec] = []
        dry_record, dry_spec = _endpoint_record(
            canonical_endpoint=None,
            endpoint_id="endpoint_000",
            role="dry",
            endpoint_mode=None,
            native_parameters=None,
            embedded_parameters=None,
            common_target_audio=selected_tensor,
            source={
                "renderer_mode": canonical.renderer_mode,
                "verification_method": "exact target selected-slice identity",
            },
        )
        endpoint_records.append(dry_record)
        endpoint_specs.append(dry_spec)

        reference_endpoints = [
            endpoint for endpoint in canonical.endpoints
            if endpoint.kind == "processed_reference"
        ]
        encoder = None
        if reference_endpoints:
            encoder = runtime_dependencies.load_encoder(
                record["reproducibility"]["reference_runtime"]
            )
            counters["encoder_loads_completed"] += 1

        routes = {
            row["endpoint_id"]: row for row in prepared_record["endpoint_materialization_routes"]
        }
        decoded_reference_snapshots: list[dict[str, Any]] = []
        for endpoint in canonical.endpoints:
            endpoint_dir = staging / "endpoints" / endpoint.endpoint_id
            endpoint_dir.mkdir(parents=True)
            route = routes[endpoint.endpoint_id]
            if endpoint.kind == "natural_language":
                request = route["request"]
                expected_native = _native_request(
                    request["payload"],
                    base_url=record["endpoint_generation_config"]["base_url"],
                    reasoning_effort=record["endpoint_generation_config"][
                        "reasoning_effort"
                    ],
                )
                if (
                    _sha_json(request["payload"]) != request["payload_sha256"]
                    or expected_native != request["native_request"]
                    or _sha_json(expected_native) != request["native_request_sha256"]
                ):
                    raise ExecuteRunError(
                        "frozen static endpoint request changed"
                    )
                counters["api_calls_started"] += 1
                raw_plan = _call_once(
                    payload=deepcopy(request["payload"]),
                    base_url=record["endpoint_generation_config"]["base_url"],
                    reasoning_effort=record["endpoint_generation_config"]["reasoning_effort"],
                    timeout=timeout,
                    output_directory=endpoint_dir,
                    plan_source=plan_source,
                    runtime_dependencies=runtime_dependency_record,
                    semantic_pre_registration=semantic_identity,
                    call_once=call_once,
                )
                counters["api_calls_completed"] += 1
                validated = runtime_dependencies.validate_static_plan(
                    raw_plan,
                    endpoint=endpoint,
                    target_stem=str(canonical.target["stem"]),
                )
                _write_json_exclusive(endpoint_dir / "validated_static_plan.json", validated)
                native_np = runtime_dependencies.materialize_static_parameters(
                    validated,
                    endpoint=endpoint,
                    target_stem=str(canonical.target["stem"]),
                    renderer_mode=None,
                )
                native_parameters = torch.as_tensor(native_np, dtype=torch.float32)
                rendered = runtime_dependencies.render_native_endpoint(
                    selected_tensor, native_parameters, endpoint.effect_mode
                )
                counters["audio_renders_completed"] += 1
                source = {
                    "renderer_mode": canonical.renderer_mode,
                    "verification_method": (
                        "exact structured request, local static schema validation, and "
                        "deterministic normalized parameter conversion"
                    ),
                    "request_artifact": _run_artifact(
                        staging, endpoint_dir / "request.json"
                    ),
                    "raw_response_artifact": _run_artifact(
                        staging, endpoint_dir / "raw_response_before_parse.json"
                    ),
                    "validated_plan_artifact": _run_artifact(
                        staging, endpoint_dir / "validated_static_plan.json"
                    ),
                    "api_calls": 1,
                    "retry": 0,
                    "repair": 0,
                }
                caller_description = endpoint.user_description
            else:
                assert endpoint.source_audio is not None and encoder is not None
                reference_audio = _load_reference_clip(endpoint.source_audio)
                if float(np.max(np.abs(reference_audio), initial=0.0)) <= 0.0:
                    raise ExecuteRunError("processed reference is silent")
                decoded_reference_snapshots.append(
                    {
                        "endpoint_id": endpoint.endpoint_id,
                        "source_audio": dict(endpoint.source_audio),
                        "decoded_stereo_sha256": _sha_array(reference_audio),
                        "frames": int(reference_audio.shape[0]),
                    }
                )
                reference_tensor = _tensor_audio(reference_audio)
                counters["optimizer_runs_started"] += 1
                match = runtime_dependencies.match_reference(
                    encoder,
                    reference_tensor,
                    selected_tensor,
                    endpoint.effect_mode,
                    renderer=runtime_dependencies.reference_renderer_class(
                        sample_rate=SAMPLE_RATE,
                        device="cpu",
                        reverb_seed=matcher.DEFAULT_REVERB_SEED,
                    ),
                    learning_rate=matcher.DEFAULT_LEARNING_RATE,
                )
                counters["optimizer_runs_completed"] += 1
                counters["optimizer_steps_completed"] += matcher.MATCH_STEPS
                if match.steps != matcher.MATCH_STEPS or len(match.loss_trace) != matcher.MATCH_STEPS + 1:
                    raise ExecuteRunError("reference optimizer step contract changed")
                native_parameters = match.normalized_parameters.detach().clone()
                rendered = match.rendered_audio.detach().clone()
                counters["audio_renders_completed"] += matcher.MATCH_STEPS + 1
                _write_json_exclusive(
                    endpoint_dir / "reference_match.json",
                    {
                        "objective": "plain Fx-Encoder++ embedding cosine",
                        "encoder_input_normalization": (
                            "differentiable per-item absolute peak with true gradient"
                        ),
                        "reference_input_count": 1,
                        "reference_dry_required": False,
                        "target_selected_stereo_sha256": selected_hash,
                        "processed_reference_file": dict(endpoint.source_audio),
                        "processed_reference_decoded_stereo_sha256": _sha_array(reference_audio),
                        "mode": endpoint.effect_mode,
                        "steps": match.steps,
                        "loss_trace": list(match.loss_trace),
                        "final_loss": float(match.loss_trace[-1]),
                        "normalized_parameters": native_parameters.tolist(),
                        "semantic_match_success_claimed": False,
                    },
                )
                source = {
                    "renderer_mode": canonical.renderer_mode,
                    "verification_method": (
                        "current target selected slice and current single processed reference "
                        "matched for exactly 60 native-mode steps"
                    ),
                    "reference_match_artifact": _run_artifact(
                        staging, endpoint_dir / "reference_match.json"
                    ),
                    "api_calls": 0,
                    "optimizer_runs": 1,
                    "optimizer_steps": matcher.MATCH_STEPS,
                    "reference_dry_required": False,
                }
                caller_description = None
            embedded = runtime_dependencies.embed_endpoint_parameters(
                native_parameters,
                endpoint_mode=endpoint.effect_mode,
                renderer_mode=canonical.renderer_mode,
            )
            endpoint_record, endpoint_spec = _endpoint_record(
                canonical_endpoint=endpoint,
                endpoint_id=endpoint.endpoint_id,
                role=(
                    "caller_natural_language"
                    if endpoint.kind == "natural_language"
                    else "processed_reference"
                ),
                endpoint_mode=endpoint.effect_mode,
                native_parameters=native_parameters,
                embedded_parameters=embedded,
                common_target_audio=rendered,
                source=source,
                caller_description_exact=caller_description,
            )
            _write_json_exclusive(endpoint_dir / "endpoint_record.json", endpoint_record)
            endpoint_records.append(endpoint_record)
            endpoint_specs.append(endpoint_spec)

        _write_json_exclusive(
            staging / "decoded_reference_snapshots.json",
            {
                "policy": "caller-selected exact 10.0-second clips; no full-song selection",
                "references": decoded_reference_snapshots,
            },
        )
        _verify_source_identities(canonical)

        registry = runtime_dependencies.registry_class(
            canonical.renderer_mode, endpoint_specs
        )
        _write_json_exclusive(staging / "endpoint_records.json", endpoint_records)
        structure_context = _structure_context(canonical)
        context = _automation_context(
            canonical=canonical,
            structure_context=structure_context,
            endpoint_records=endpoint_records,
        )
        payload = _automation_payload(record=record, context=context)
        automation_dir = staging / "automation"
        automation_dir.mkdir()
        _write_json_exclusive(automation_dir / "llm_safe_context.json", context)
        counters["api_calls_started"] += 1
        raw_plan = _call_once(
            payload=payload,
            base_url=record["endpoint_generation_config"]["base_url"],
            reasoning_effort=record["endpoint_generation_config"]["reasoning_effort"],
            timeout=timeout,
            output_directory=automation_dir,
            plan_source=plan_source,
            runtime_dependencies=runtime_dependency_record,
            semantic_pre_registration=semantic_identity,
            call_once=call_once,
        )
        counters["api_calls_completed"] += 1
        validated_plan = runtime_dependencies.validate_automation_plan(
            raw_plan,
            schema=record["automation_schema"],
            registry=registry,
            instruction=canonical.automation_instruction,
        )
        _write_json_exclusive(automation_dir / "validated_plan.json", validated_plan)

        render_source = _render_source(target, record["timing"])
        _safe_audio(render_source, "target render source")
        render_tensor = _tensor_audio(render_source)
        renderer = runtime_dependencies.final_renderer_class(sample_rate=SAMPLE_RATE)
        processed, render_report = renderer.render_plan(
            render_tensor,
            registry=registry,
            validated_plan=validated_plan,
            plan_schema=record["automation_schema"],
            instruction=canonical.automation_instruction,
            active_start_sample=int(record["timing"]["active_start_sample_relative"]),
            active_stop_sample=int(record["timing"]["active_stop_sample_relative"]),
            attack_start_sample=int(record["timing"]["attack_start_sample_relative"]),
            release_stop_sample=int(record["timing"]["release_stop_sample_relative"]),
            require_complete_reverb_tail=True,
            require_safe_output=True,
        )
        counters["audio_renders_completed"] += 1
        processed_np = np.ascontiguousarray(processed[0].T.detach().cpu().numpy())
        after_safety = _safe_audio(processed_np, "final automation output")

        active_samples = int(record["timing"]["active_stop_sample_relative"]) - int(
            record["timing"]["active_start_sample_relative"]
        )
        core_weights, schedule = planner.evaluate_endpoint_schedule(
            validated_plan, registry=registry, n_samples=active_samples
        )
        full_weights = renderer_module.extend_endpoint_weights(
            core_weights,
            total_samples=render_source.shape[0],
            active_start_sample=int(record["timing"]["active_start_sample_relative"]),
            active_stop_sample=int(record["timing"]["active_stop_sample_relative"]),
        )
        activation = renderer_module.build_boundary_activation(
            render_source.shape[0],
            active_start_sample=int(record["timing"]["active_start_sample_relative"]),
            active_stop_sample=int(record["timing"]["active_stop_sample_relative"]),
            attack_start_sample=int(record["timing"]["attack_start_sample_relative"]),
            release_stop_sample=int(record["timing"]["release_stop_sample_relative"]),
        )
        if (
            _sha_array(full_weights) != render_report["endpoint_weights_sha256_after"]
            or _sha_array(activation) != render_report["activation_sha256_after"]
        ):
            raise ExecuteRunError("saved control reconstruction changed")
        np.save(staging / "resolved_endpoint_weights.npy", full_weights, allow_pickle=False)
        np.save(staging / "boundary_activation.npy", activation, allow_pickle=False)
        _write_json_exclusive(
            staging / "resolved_controls.json",
            {
                "endpoint_ids": list(registry.endpoint_ids),
                "schedule": schedule,
                "full_endpoint_weights_sha256": _sha_array(full_weights),
                "boundary_activation_sha256": _sha_array(activation),
            },
        )
        _write_float(staging / "target_before_FLOAT.wav", render_source)
        _write_float(staging / "target_after_FLOAT.wav", processed_np)
        _verify_source_identities(canonical)

        budget = record["exact_work_budget"]
        if (
            counters["api_calls_completed"] != int(budget["total_api_calls"])
            or counters["optimizer_runs_completed"] != int(budget["reference_optimizer_runs"])
            or counters["optimizer_steps_completed"] != int(
                budget["reference_optimizer_steps_total"]
            )
            or counters["encoder_loads_completed"] != int(budget["encoder_loads"])
            or counters["audio_renders_completed"] != int(
                budget["total_audio_renders"]
            )
        ):
            raise ExecuteRunError("actual work counts differ from preregistration")
        if plan_source == LIVE_PLAN_SOURCE:
            if _transport_record(plan_source) != transport:
                raise ExecuteRunError(
                    "live transport changed during execution"
                )
            if (
                _validate_runtime_dependency_bundle(
                    plan_source, runtime_dependencies
                )
                != runtime_dependency_record
            ):
                raise ExecuteRunError(
                    "live runtime dependencies changed during execution"
                )
        if _reverify_semantic_identity(semantic_identity) != semantic_digest:
            raise ExecuteRunError(
                "semantic preregistration changed during execution"
            )
        artifact_manifest = _artifact_manifest(staging)
        result = {
            "run_version": run_version,
            "plan_source": plan_source,
            "transport": transport,
            "runtime_dependencies": runtime_dependency_record,
            "semantic_pre_registration_sha256": semantic_digest,
            "status": "completed",
            "pre_registration_sha256": prereg_digest,
            "canonical_manifest_sha256": record["canonical_manifest_sha256"],
            "source_audio_array_sha256": _sha_array(render_source),
            "processed_audio_array_sha256": _sha_array(processed_np),
            "work_counters": counters,
            "endpoint_records": endpoint_records,
            "automation": {
                "context_sha256": _sha_json(context),
                "payload_sha256": _sha_json(payload),
                "raw_plan": raw_plan,
                "validated_plan": validated_plan,
                "api_calls": 1,
                "retry": 0,
                "repair": 0,
            },
            "timing": record["timing"],
            "render": {
                "schedule": schedule,
                "diagnostics": render_report,
                "endpoint_weights_sha256": _sha_array(full_weights),
                "boundary_activation_sha256": _sha_array(activation),
                "output_safety": after_safety,
            },
            "artifacts": {
                "endpoint_records": _run_artifact(
                    staging, staging / "endpoint_records.json"
                ),
                "automation_context": _run_artifact(
                    staging, automation_dir / "llm_safe_context.json"
                ),
                "automation_request": _run_artifact(
                    staging, automation_dir / "request.json"
                ),
                "automation_raw_response": _run_artifact(
                    staging, automation_dir / "raw_response_before_parse.json"
                ),
                "automation_raw_plan": _run_artifact(
                    staging, automation_dir / "raw_plan.json"
                ),
                "automation_validated_plan": _run_artifact(
                    staging, automation_dir / "validated_plan.json"
                ),
                "source_snapshot": _run_artifact(
                    staging, staging / "source_snapshot.json"
                ),
                "decoded_reference_snapshots": _run_artifact(
                    staging, staging / "decoded_reference_snapshots.json"
                ),
                "resolved_controls": _run_artifact(
                    staging, staging / "resolved_controls.json"
                ),
                "resolved_endpoint_weights": _run_artifact(
                    staging, staging / "resolved_endpoint_weights.npy"
                ),
                "boundary_activation": _run_artifact(
                    staging, staging / "boundary_activation.npy"
                ),
                "target_before": _run_artifact(
                    staging, staging / "target_before_FLOAT.wav"
                ),
                "target_after": _run_artifact(
                    staging, staging / "target_after_FLOAT.wav"
                ),
                "semantic_pre_registration": _run_artifact(
                    staging,
                    staging
                    / "semantic_pre_registration"
                    / "SEMANTIC_PRE_REGISTRATION.json",
                ),
                "semantic_pre_registration_sidecar": _run_artifact(
                    staging,
                    staging
                    / "semantic_pre_registration"
                    / "SEMANTIC_PRE_REGISTRATION.sha256",
                ),
                "semantic_preparation_status": _run_artifact(
                    staging,
                    staging
                    / "semantic_pre_registration"
                    / "PREPARATION_STATUS.json",
                ),
            },
            "artifact_manifest": artifact_manifest,
            "claims": {
                "semantic_style_match_success": False,
                "llm_heard_audio": False,
                "reference_dry_used": False,
                "limiter_or_hidden_repair": False,
                "full_mixture_remixed": False,
            },
            "elapsed_sec": time.perf_counter() - started,
        }
        _write_json_exclusive(staging / "result.json", result)
        _write_json_exclusive(
            staging / "COMPLETED.json",
            {
                "status": "completed",
                "run_version": run_version,
                "plan_source": plan_source,
                "transport": transport,
                "runtime_dependencies": runtime_dependency_record,
                "semantic_pre_registration_sha256": semantic_digest,
                "result_sha256": manifest.file_sha256(staging / "result.json"),
                "api_calls": counters["api_calls_completed"],
                "optimizer_runs": counters["optimizer_runs_completed"],
                "optimizer_steps": counters["optimizer_steps_completed"],
                "encoder_loads": counters["encoder_loads_completed"],
                "audio_renders": counters["audio_renders_completed"],
            },
        )
    except Exception as error:
        for incompatible in (staging / "result.json", staging / "COMPLETED.json"):
            incompatible.unlink(missing_ok=True)
        _write_json_exclusive(
            staging / "FAILURE.json",
            {
                "status": "failed",
                "run_version": run_version,
                "plan_source": plan_source,
                "transport": transport,
                "runtime_dependencies": runtime_dependency_record,
                "error_type": type(error).__name__,
                "error": str(error),
                "work_counters": counters,
                "retry": 0,
                "repair": 0,
                "traceback": traceback.format_exc(),
                "elapsed_sec": time.perf_counter() - started,
            },
        )
        os.replace(staging, output)
        raise
    os.replace(staging, output)
    return output


def execute(
    *,
    prepared: Path,
    output: Path,
    reviewed: bool,
    timeout: float = 900.0,
) -> Path:
    return _execute(
        prepared=prepared,
        output=output,
        reviewed=reviewed,
        plan_source=LIVE_PLAN_SOURCE,
        call_once=_LIVE_OPENAI_CALL,
        runtime_dependencies=_live_runtime_dependencies(),
        timeout=timeout,
    )


def _execute_for_test(
    *,
    prepared: Path,
    output: Path,
    reviewed: bool,
    test_double: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
    runtime_dependencies: _RuntimeDependencies,
    timeout: float = 900.0,
) -> Path:
    if test_double is _LIVE_OPENAI_CALL:
        raise ExecuteRunError(
            "test entry requires an explicit non-live test double"
        )
    return _execute(
        prepared=prepared,
        output=output,
        reviewed=reviewed,
        plan_source=TEST_PLAN_SOURCE,
        call_once=test_double,
        runtime_dependencies=runtime_dependencies,
        timeout=timeout,
    )


def _verify_completed_run(
    path: Path, *, expected_plan_source: str
) -> dict[str, Any]:
    expected_run_version = _run_version(expected_plan_source)
    expected_transport = _transport_record(expected_plan_source)
    expected_runtime_dependencies = _runtime_dependency_record(
        expected_plan_source
    )
    replay_dependencies = _live_runtime_dependencies()
    path = path.resolve()
    if not path.is_dir() or (path / "FAILURE.json").exists():
        raise ExecuteRunError("completed run is missing or has FAILURE")
    for required in (
        "PRE_REGISTRATION.json",
        "PRE_REGISTRATION.sha256",
        "result.json",
        "COMPLETED.json",
        "resolved_controls.json",
        "resolved_endpoint_weights.npy",
        "boundary_activation.npy",
        "target_before_FLOAT.wav",
        "target_after_FLOAT.wav",
    ):
        if not (path / required).is_file():
            raise ExecuteRunError(f"completed artifact missing: {required}")
    prereg = _read_json(path / "PRE_REGISTRATION.json")
    digest = manifest.file_sha256(path / "PRE_REGISTRATION.json")
    if path.joinpath("PRE_REGISTRATION.sha256").read_text(
        encoding="ascii"
    ).split()[0] != digest:
        raise ExecuteRunError("completed preregistration hash changed")
    if (
        prereg.get("run_version") != expected_run_version
        or prereg.get("plan_source") != expected_plan_source
        or prereg.get("transport") != expected_transport
        or prereg.get("runtime_dependencies") != expected_runtime_dependencies
    ):
        raise ExecuteRunError("completed preregistration version changed")
    semantic_identity = prereg.get("semantic_evaluation_pre_registration")
    semantic_digest = _reverify_semantic_identity(semantic_identity)
    if prereg.get("semantic_pre_registration_sha256") != semantic_digest:
        raise ExecuteRunError(
            "completed semantic preregistration binding changed"
        )
    for name, identity in prereg["reproducibility"]["code"].items():
        if _identity(Path(identity["path"])) != identity:
            raise ExecuteRunError(f"completed code changed: {name}")
    if _build_execution_pre_registration(
        prepared_path=Path(prereg["source_v3_prepared"]["directory"]),
        semantic_prepared=Path(semantic_identity["directory"]),
        plan_source=expected_plan_source,
    ) != prereg:
        raise ExecuteRunError(
            "completed preregistration no longer matches frozen inputs"
        )
    result = _read_json(path / "result.json")
    completed = _read_json(path / "COMPLETED.json")
    expected_result_fields = {
        "run_version",
        "plan_source",
        "transport",
        "runtime_dependencies",
        "semantic_pre_registration_sha256",
        "status",
        "pre_registration_sha256",
        "canonical_manifest_sha256",
        "source_audio_array_sha256",
        "processed_audio_array_sha256",
        "work_counters",
        "endpoint_records",
        "automation",
        "timing",
        "render",
        "artifacts",
        "artifact_manifest",
        "claims",
        "elapsed_sec",
    }
    expected_completed_fields = {
        "status",
        "run_version",
        "plan_source",
        "transport",
        "runtime_dependencies",
        "semantic_pre_registration_sha256",
        "result_sha256",
        "api_calls",
        "optimizer_runs",
        "optimizer_steps",
        "encoder_loads",
        "audio_renders",
    }
    if (
        set(result) != expected_result_fields
        or set(completed) != expected_completed_fields
        or result.get("status") != "completed"
        or result.get("run_version") != expected_run_version
        or result.get("plan_source") != expected_plan_source
        or result.get("transport") != expected_transport
        or result.get("runtime_dependencies") != expected_runtime_dependencies
        or result.get("semantic_pre_registration_sha256") != semantic_digest
        or result.get("pre_registration_sha256") != digest
        or result.get("timing") != prereg["timing"]
        or result.get("claims")
        != {
            "semantic_style_match_success": False,
            "llm_heard_audio": False,
            "reference_dry_used": False,
            "limiter_or_hidden_repair": False,
            "full_mixture_remixed": False,
        }
        or isinstance(result.get("elapsed_sec"), bool)
        or not isinstance(result.get("elapsed_sec"), (int, float))
        or not np.isfinite(float(result["elapsed_sec"]))
        or float(result["elapsed_sec"]) < 0.0
        or completed.get("status") != "completed"
        or completed.get("run_version") != expected_run_version
        or completed.get("plan_source") != expected_plan_source
        or completed.get("transport") != expected_transport
        or completed.get("runtime_dependencies") != expected_runtime_dependencies
        or completed.get("semantic_pre_registration_sha256") != semantic_digest
        or completed.get("result_sha256") != manifest.file_sha256(path / "result.json")
        or completed.get("api_calls") != result["work_counters"].get(
            "api_calls_completed"
        )
        or completed.get("optimizer_runs") != result["work_counters"].get(
            "optimizer_runs_completed"
        )
        or completed.get("optimizer_steps") != result["work_counters"].get(
            "optimizer_steps_completed"
        )
        or completed.get("encoder_loads") != result["work_counters"].get(
            "encoder_loads_completed"
        )
        or completed.get("audio_renders") != result["work_counters"].get(
            "audio_renders_completed"
        )
    ):
        raise ExecuteRunError("completion markers disagree")
    if set(result["artifacts"]) != {
        "endpoint_records",
        "automation_context",
        "automation_request",
        "automation_raw_response",
        "automation_raw_plan",
        "automation_validated_plan",
        "source_snapshot",
        "decoded_reference_snapshots",
        "resolved_controls",
        "resolved_endpoint_weights",
        "boundary_activation",
        "target_before",
        "target_after",
        "semantic_pre_registration",
        "semantic_pre_registration_sidecar",
        "semantic_preparation_status",
    }:
        raise ExecuteRunError("result artifact index fields changed")
    if set(result["work_counters"]) != {
        "api_calls_started",
        "api_calls_completed",
        "optimizer_runs_started",
        "optimizer_runs_completed",
        "optimizer_steps_completed",
        "encoder_loads_completed",
        "audio_renders_completed",
    }:
        raise ExecuteRunError("result work-counter fields changed")
    if set(result["render"]) != {
        "schedule",
        "diagnostics",
        "endpoint_weights_sha256",
        "boundary_activation_sha256",
        "output_safety",
    }:
        raise ExecuteRunError("result render fields changed")
    expected_artifacts = {
        "endpoint_records": _run_artifact(path, path / "endpoint_records.json"),
        "automation_context": _run_artifact(
            path, path / "automation" / "llm_safe_context.json"
        ),
        "automation_request": _run_artifact(
            path, path / "automation" / "request.json"
        ),
        "automation_raw_response": _run_artifact(
            path, path / "automation" / "raw_response_before_parse.json"
        ),
        "automation_raw_plan": _run_artifact(
            path, path / "automation" / "raw_plan.json"
        ),
        "automation_validated_plan": _run_artifact(
            path, path / "automation" / "validated_plan.json"
        ),
        "source_snapshot": _run_artifact(path, path / "source_snapshot.json"),
        "decoded_reference_snapshots": _run_artifact(
            path, path / "decoded_reference_snapshots.json"
        ),
        "resolved_controls": _run_artifact(path, path / "resolved_controls.json"),
        "resolved_endpoint_weights": _run_artifact(
            path, path / "resolved_endpoint_weights.npy"
        ),
        "boundary_activation": _run_artifact(
            path, path / "boundary_activation.npy"
        ),
        "target_before": _run_artifact(path, path / "target_before_FLOAT.wav"),
        "target_after": _run_artifact(path, path / "target_after_FLOAT.wav"),
        "semantic_pre_registration": _run_artifact(
            path,
            path
            / "semantic_pre_registration"
            / "SEMANTIC_PRE_REGISTRATION.json",
        ),
        "semantic_pre_registration_sidecar": _run_artifact(
            path,
            path
            / "semantic_pre_registration"
            / "SEMANTIC_PRE_REGISTRATION.sha256",
        ),
        "semantic_preparation_status": _run_artifact(
            path,
            path / "semantic_pre_registration" / "PREPARATION_STATUS.json",
        ),
    }
    if result["artifacts"] != expected_artifacts:
        raise ExecuteRunError("result artifact label binding changed")
    semantic_copy = path / "semantic_pre_registration"
    for copied_name, identity_key in (
        ("SEMANTIC_PRE_REGISTRATION.json", "pre_registration"),
        ("SEMANTIC_PRE_REGISTRATION.sha256", "sidecar"),
        ("PREPARATION_STATUS.json", "preparation_status"),
    ):
        if (
            manifest.file_sha256(semantic_copy / copied_name)
            != semantic_identity[identity_key]["sha256"]
        ):
            raise ExecuteRunError(
                "copied semantic preregistration bundle changed"
            )

    saved_manifest = result.get("artifact_manifest")
    if not isinstance(saved_manifest, Mapping) or dict(saved_manifest) != _artifact_manifest(path):
        raise ExecuteRunError(
            "completed recursive artifact manifest changed"
        )
    expected_files = set(saved_manifest) | {"result.json", "COMPLETED.json"}
    actual_files = {
        artifact.relative_to(path).as_posix()
        for artifact in path.rglob("*")
        if artifact.is_file()
    }
    if actual_files != expected_files:
        raise ExecuteRunError(
            "completed run contains missing or unregistered artifacts"
        )
    for identity in result["artifacts"].values():
        _verify_run_artifact(path, identity)

    prepared_record, _ = prepare_run.verify_prepared(
        Path(prereg["source_v3_prepared"]["directory"])
    )
    canonical = _canonical_manifest(prepared_record)
    if _sha_json(canonical.to_dict()) != result["canonical_manifest_sha256"]:
        raise ExecuteRunError("completed canonical manifest changed")
    _verify_source_identities(canonical)
    target = _load_audio(canonical.target["raw_stem"])
    timing = prereg["timing"]
    selected = target[
        int(timing["selected_source_start_sample"]) :
        int(timing["selected_source_stop_sample_exclusive"])
    ]
    selected_hash = _sha_array(selected)
    if selected_hash != canonical.target["selected_section_audio"][
        "decoded_stereo_sha256"
    ]:
        raise ExecuteRunError("completed target slice changed")
    selected_tensor = _tensor_audio(selected)
    expected_source_snapshot = {
        "manifest": dict(canonical.source_manifest),
        "target_file": dict(canonical.target["raw_stem"]),
        "structure_file": {
            key: canonical.target["structure"][key]
            for key in ("path", "sha256")
        },
        "target_decoded_stereo_sha256": _sha_array(target),
        "target_selected_stereo_sha256": selected_hash,
        "reference_files": [
            dict(endpoint.source_audio)
            for endpoint in canonical.endpoints
            if endpoint.source_audio is not None
        ],
        "processed_reference_policy": {
            "caller_selected_precropped_clip": True,
            "required_frames": REFERENCE_CLIP_FRAMES,
            "required_duration_sec": 10.0,
            "full_song_selection_performed": False,
        },
    }
    if _read_json(path / "source_snapshot.json") != expected_source_snapshot:
        raise ExecuteRunError("completed source snapshot changed")

    routes = {
        row["endpoint_id"]: row
        for row in prepared_record["endpoint_materialization_routes"]
    }
    saved_endpoint_records = _read_json_value(path / "endpoint_records.json")
    if not isinstance(saved_endpoint_records, list):
        raise ExecuteRunError("endpoint records are not a list")
    reconstructed_records: list[dict[str, Any]] = []
    reconstructed_specs: list[masked_runtime.MaskedEndpointSpec] = []
    dry_record, dry_spec = _endpoint_record(
        canonical_endpoint=None,
        endpoint_id="endpoint_000",
        role="dry",
        endpoint_mode=None,
        native_parameters=None,
        embedded_parameters=None,
        common_target_audio=selected_tensor,
        source={
            "renderer_mode": canonical.renderer_mode,
            "verification_method": "exact target selected-slice identity",
        },
    )
    reconstructed_records.append(dry_record)
    reconstructed_specs.append(dry_spec)
    decoded_references: list[dict[str, Any]] = []
    config = prereg["endpoint_generation_config"]
    for endpoint in canonical.endpoints:
        endpoint_dir = path / "endpoints" / endpoint.endpoint_id
        saved_record = _read_json(endpoint_dir / "endpoint_record.json")
        route = routes[endpoint.endpoint_id]
        if endpoint.kind == "natural_language":
            raw = _verify_saved_call(
                run_root=path,
                call_directory=endpoint_dir,
                expected_payload=route["request"]["payload"],
                base_url=config["base_url"],
                reasoning_effort=config["reasoning_effort"],
                expected_plan_source=expected_plan_source,
                expected_runtime_dependencies=expected_runtime_dependencies,
                expected_semantic_pre_registration=semantic_identity,
            )
            validated = replay_dependencies.validate_static_plan(
                raw,
                endpoint=endpoint,
                target_stem=str(canonical.target["stem"]),
            )
            if _read_json(endpoint_dir / "validated_static_plan.json") != validated:
                raise ExecuteRunError(
                    "saved static endpoint validation changed"
                )
            native_np = replay_dependencies.materialize_static_parameters(
                validated,
                endpoint=endpoint,
                target_stem=str(canonical.target["stem"]),
                renderer_mode=None,
            )
            native_parameters = torch.as_tensor(native_np, dtype=torch.float32)
            common_target = replay_dependencies.render_native_endpoint(
                selected_tensor, native_parameters, endpoint.effect_mode
            )
            source = {
                "renderer_mode": canonical.renderer_mode,
                "verification_method": (
                    "exact structured request, local static schema validation, and "
                    "deterministic normalized parameter conversion"
                ),
                "request_artifact": _run_artifact(
                    path, endpoint_dir / "request.json"
                ),
                "raw_response_artifact": _run_artifact(
                    path, endpoint_dir / "raw_response_before_parse.json"
                ),
                "validated_plan_artifact": _run_artifact(
                    path, endpoint_dir / "validated_static_plan.json"
                ),
                "api_calls": 1,
                "retry": 0,
                "repair": 0,
            }
            role = "caller_natural_language"
            caller_description = endpoint.user_description
        else:
            if endpoint.source_audio is None:
                raise ExecuteRunError(
                    "processed-reference source disappeared"
                )
            reference_audio = _load_reference_clip(endpoint.source_audio)
            reference_match = _read_json(endpoint_dir / "reference_match.json")
            required_match_fields = {
                "objective",
                "encoder_input_normalization",
                "reference_input_count",
                "reference_dry_required",
                "target_selected_stereo_sha256",
                "processed_reference_file",
                "processed_reference_decoded_stereo_sha256",
                "mode",
                "steps",
                "loss_trace",
                "final_loss",
                "normalized_parameters",
                "semantic_match_success_claimed",
            }
            losses = reference_match.get("loss_trace")
            if (
                set(reference_match) != required_match_fields
                or reference_match.get("objective")
                != "plain Fx-Encoder++ embedding cosine"
                or reference_match.get("encoder_input_normalization")
                != "differentiable per-item absolute peak with true gradient"
                or reference_match.get("reference_input_count") != 1
                or reference_match.get("reference_dry_required") is not False
                or reference_match.get("target_selected_stereo_sha256") != selected_hash
                or reference_match.get("processed_reference_file")
                != dict(endpoint.source_audio)
                or reference_match.get("processed_reference_decoded_stereo_sha256")
                != _sha_array(reference_audio)
                or reference_match.get("mode") != endpoint.effect_mode
                or reference_match.get("steps") != matcher.MATCH_STEPS
                or not isinstance(losses, list)
                or len(losses) != matcher.MATCH_STEPS + 1
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and np.isfinite(float(value))
                    for value in losses
                )
                or float(reference_match.get("final_loss")) != float(losses[-1])
                or reference_match.get("semantic_match_success_claimed") is not False
            ):
                raise ExecuteRunError(
                    "saved processed-reference match record changed"
                )
            native_parameters = torch.as_tensor(
                reference_match["normalized_parameters"], dtype=torch.float32
            )
            common_target = replay_dependencies.render_native_endpoint(
                selected_tensor, native_parameters, endpoint.effect_mode
            )
            decoded_references.append(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "source_audio": dict(endpoint.source_audio),
                    "decoded_stereo_sha256": _sha_array(reference_audio),
                    "frames": int(reference_audio.shape[0]),
                }
            )
            source = {
                "renderer_mode": canonical.renderer_mode,
                "verification_method": (
                    "current target selected slice and current single processed reference "
                    "matched for exactly 60 native-mode steps"
                ),
                "reference_match_artifact": _run_artifact(
                    path, endpoint_dir / "reference_match.json"
                ),
                "api_calls": 0,
                "optimizer_runs": 1,
                "optimizer_steps": matcher.MATCH_STEPS,
                "reference_dry_required": False,
            }
            role = "processed_reference"
            caller_description = None
        embedded = replay_dependencies.embed_endpoint_parameters(
            native_parameters,
            endpoint_mode=endpoint.effect_mode,
            renderer_mode=canonical.renderer_mode,
        )
        expected_record, expected_spec = _endpoint_record(
            canonical_endpoint=endpoint,
            endpoint_id=endpoint.endpoint_id,
            role=role,
            endpoint_mode=endpoint.effect_mode,
            native_parameters=native_parameters,
            embedded_parameters=embedded,
            common_target_audio=common_target,
            source=source,
            caller_description_exact=caller_description,
        )
        if saved_record != expected_record:
            raise ExecuteRunError(
                f"completed endpoint replay changed: {endpoint.endpoint_id}"
            )
        reconstructed_records.append(expected_record)
        reconstructed_specs.append(expected_spec)
    expected_reference_snapshots = {
        "policy": "caller-selected exact 10.0-second clips; no full-song selection",
        "references": decoded_references,
    }
    if _read_json(
        path / "decoded_reference_snapshots.json"
    ) != expected_reference_snapshots:
        raise ExecuteRunError(
            "decoded processed-reference snapshots changed"
        )
    if (
        reconstructed_records != saved_endpoint_records
        or reconstructed_records != result["endpoint_records"]
    ):
        raise ExecuteRunError("completed endpoint registry changed")

    registry = replay_dependencies.registry_class(
        canonical.renderer_mode, reconstructed_specs
    )
    structure_context = _structure_context(canonical)
    context = _automation_context(
        canonical=canonical,
        structure_context=structure_context,
        endpoint_records=reconstructed_records,
    )
    if _read_json(path / "automation" / "llm_safe_context.json") != context:
        raise ExecuteRunError("saved automation context changed")
    payload = _automation_payload(record=prereg, context=context)
    raw_plan = _verify_saved_call(
        run_root=path,
        call_directory=path / "automation",
        expected_payload=payload,
        base_url=config["base_url"],
        reasoning_effort=config["reasoning_effort"],
        expected_plan_source=expected_plan_source,
        expected_runtime_dependencies=expected_runtime_dependencies,
        expected_semantic_pre_registration=semantic_identity,
    )
    validated_plan = replay_dependencies.validate_automation_plan(
        raw_plan,
        schema=prereg["automation_schema"],
        registry=registry,
        instruction=canonical.automation_instruction,
    )
    if (
        _read_json(path / "automation" / "validated_plan.json") != validated_plan
        or result["automation"] != {
            "context_sha256": _sha_json(context),
            "payload_sha256": _sha_json(payload),
            "raw_plan": raw_plan,
            "validated_plan": validated_plan,
            "api_calls": 1,
            "retry": 0,
            "repair": 0,
        }
    ):
        raise ExecuteRunError("automation response chain changed")

    weights = np.load(path / "resolved_endpoint_weights.npy", allow_pickle=False)
    activation = np.load(path / "boundary_activation.npy", allow_pickle=False)
    controls = _read_json(path / "resolved_controls.json")
    if set(controls) != {
        "endpoint_ids",
        "schedule",
        "full_endpoint_weights_sha256",
        "boundary_activation_sha256",
    }:
        raise ExecuteRunError("resolved-control fields changed")
    render_source = _render_source(target, timing)
    active_samples = int(timing["active_stop_sample_relative"]) - int(
        timing["active_start_sample_relative"]
    )
    core_weights, schedule = planner.evaluate_endpoint_schedule(
        validated_plan, registry=registry, n_samples=active_samples
    )
    expected_weights = renderer_module.extend_endpoint_weights(
        core_weights,
        total_samples=render_source.shape[0],
        active_start_sample=int(timing["active_start_sample_relative"]),
        active_stop_sample=int(timing["active_stop_sample_relative"]),
    )
    expected_activation = renderer_module.build_boundary_activation(
        render_source.shape[0],
        active_start_sample=int(timing["active_start_sample_relative"]),
        active_stop_sample=int(timing["active_stop_sample_relative"]),
        attack_start_sample=int(timing["attack_start_sample_relative"]),
        release_stop_sample=int(timing["release_stop_sample_relative"]),
    )
    if (
        not np.array_equal(weights, expected_weights)
        or not np.array_equal(activation, expected_activation)
        or controls["endpoint_ids"] != list(registry.endpoint_ids)
        or controls["schedule"] != schedule
        or _sha_array(weights) != controls["full_endpoint_weights_sha256"]
        or _sha_array(activation) != controls["boundary_activation_sha256"]
        or _sha_array(weights) != result["render"]["endpoint_weights_sha256"]
        or _sha_array(activation) != result["render"]["boundary_activation_sha256"]
    ):
        raise ExecuteRunError("completed control hashes changed")

    processed, render_report = replay_dependencies.final_renderer_class(
        sample_rate=SAMPLE_RATE
    ).render_plan(
        _tensor_audio(render_source),
        registry=registry,
        validated_plan=validated_plan,
        plan_schema=prereg["automation_schema"],
        instruction=canonical.automation_instruction,
        active_start_sample=int(timing["active_start_sample_relative"]),
        active_stop_sample=int(timing["active_stop_sample_relative"]),
        attack_start_sample=int(timing["attack_start_sample_relative"]),
        release_stop_sample=int(timing["release_stop_sample_relative"]),
        require_complete_reverb_tail=True,
        require_safe_output=True,
    )
    processed_np = np.ascontiguousarray(processed[0].T.detach().cpu().numpy())
    output_safety = _safe_audio(processed_np, "verified final output")
    if (
        result["render"]["schedule"] != schedule
        or result["render"]["diagnostics"] != render_report
        or result["render"]["output_safety"] != output_safety
    ):
        raise ExecuteRunError("completed renderer report changed")
    before, before_sr = sf.read(
        path / "target_before_FLOAT.wav", dtype="float32", always_2d=True
    )
    after, after_sr = sf.read(
        path / "target_after_FLOAT.wav", dtype="float32", always_2d=True
    )
    if before_sr != SAMPLE_RATE or after_sr != SAMPLE_RATE:
        raise ExecuteRunError("completed audio sample rate changed")
    if (
        not np.array_equal(before, render_source)
        or not np.array_equal(after, processed_np)
        or _sha_array(before) != result["source_audio_array_sha256"]
        or _sha_array(after) != result["processed_audio_array_sha256"]
    ):
        raise ExecuteRunError("completed decoded audio hash changed")
    budget = prereg["exact_work_budget"]
    counters = result["work_counters"]
    if (
        counters["api_calls_started"] != budget["total_api_calls"]
        or counters["api_calls_completed"] != budget["total_api_calls"]
        or counters["optimizer_runs_started"] != budget["reference_optimizer_runs"]
        or counters["optimizer_runs_completed"] != budget["reference_optimizer_runs"]
        or counters["optimizer_steps_completed"] != budget[
            "reference_optimizer_steps_total"
        ]
        or counters["encoder_loads_completed"] != budget["encoder_loads"]
        or counters["audio_renders_completed"] != budget["total_audio_renders"]
    ):
        raise ExecuteRunError("completed work counters changed")
    return {
        "verified": True,
        "run_version": expected_run_version,
        "plan_source": expected_plan_source,
        "pre_registration_sha256": digest,
        "result_sha256": manifest.file_sha256(path / "result.json"),
        "artifact_checks": {
            "completion_markers": True,
            "plan_and_runtime_source_binding": True,
            "pre_api_semantic_preregistration": True,
            "code_identities": True,
            "recursive_artifact_manifest": True,
            "source_snapshot_and_toctou": True,
            "raw_response_to_validated_plan": True,
            "endpoint_common_target_replay": True,
            "automation_context_and_payload": True,
            "control_hashes": True,
            "renderer_replay": True,
            "decoded_audio_hashes": True,
            "work_counters": True,
        },
    }


def verify_completed_run(path: Path) -> dict[str, Any]:
    return _verify_completed_run(path, expected_plan_source=LIVE_PLAN_SOURCE)


def _verify_completed_test_run(path: Path) -> dict[str, Any]:
    return _verify_completed_run(path, expected_plan_source=TEST_PLAN_SOURCE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-execution")
    prepare.add_argument("--prepared-run", dest="prepared_path", type=Path, required=True)
    prepare.add_argument("--semantic-case", dest="semantic_prereg", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reviewed", action="store_true")
    run.add_argument("--timeout", type=float, default=900.0)
    verify = commands.add_parser("verify-completed")
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-execution":
        result: Mapping[str, Any] = {
            "output": str(
                prepare_execution(
                    prepared_path=args.prepared_path,
                    semantic_prepared=args.semantic_prereg,
                    output=args.output,
                )
            )
        }
    elif args.command == "execute":
        result = {
            "output": str(
                execute(
                    prepared=args.prepared,
                    output=args.output,
                    reviewed=args.reviewed,
                    timeout=args.timeout,
                )
            )
        }
    else:
        result = verify_completed_run(args.run)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
