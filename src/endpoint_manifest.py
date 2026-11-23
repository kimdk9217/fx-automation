"""Input manifest: target stem, section, instruction and endpoint definitions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

SCHEMA_VERSION = "dynamic-user-endpoint-manifest-v3"
SAMPLE_RATE = 44_100
TARGET_STEMS = ("vocals", "drums", "bass", "other")
EFFECT_MODES = ("eq", "reverb", "eq_reverb")
ENDPOINT_KINDS = ("natural_language", "processed_reference")
MAX_USER_ENDPOINTS = 99
_SECTION_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
_INTERNAL_ENDPOINT_TOKEN = re.compile(r"endpoint_[0-9]{3}", re.I)


class DynamicEndpointManifestError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalEndpoint:
    endpoint_id: str
    kind: str
    effect_mode: str
    user_description: str | None
    source_audio: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "endpoint_id": self.endpoint_id,
            "identifier_visibility": "hidden_internal_identifier",
            "kind": self.kind,
            "effect_mode": self.effect_mode,
        }
        if self.user_description is not None:
            result["user_description"] = self.user_description
            result["user_description_utf8_sha256"] = hashlib.sha256(
                self.user_description.encode("utf-8")
            ).hexdigest()
        if self.source_audio is not None:
            result["source_audio"] = dict(self.source_audio)
        return result


@dataclass(frozen=True)
class CanonicalManifest:
    source_manifest: Mapping[str, str]
    target: Mapping[str, Any]
    automation_instruction: str
    renderer_mode: str
    endpoints: tuple[CanonicalEndpoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_manifest": dict(self.source_manifest),
            "target": dict(self.target),
            "automation_instruction": self.automation_instruction,
            "automation_instruction_utf8_sha256": hashlib.sha256(
                self.automation_instruction.encode("utf-8")
            ).hexdigest(),
            "renderer_mode": self.renderer_mode,
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise DynamicEndpointManifestError(f"source file does not exist: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _exact_object(value: Any, fields: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicEndpointManifestError(f"{where} must be an object")
    result = dict(value)
    if set(result) != fields:
        raise DynamicEndpointManifestError(
            f"{where} fields must be exactly {sorted(fields)}"
        )
    return result


def _exact_text(value: Any, where: str, *, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicEndpointManifestError(f"{where} must be non-empty text")
    if len(value) > maximum or "\x00" in value:
        raise DynamicEndpointManifestError(f"{where} is too long or contains NUL")
    if _INTERNAL_ENDPOINT_TOKEN.search(value):
        raise DynamicEndpointManifestError(
            f"{where} cannot name a hidden internal endpoint identifier"
        )
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DynamicEndpointManifestError(f"{where} must be a positive integer")
    return int(value)


def _resolve_path(value: Any, base_dir: Path, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DynamicEndpointManifestError(f"{where} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise DynamicEndpointManifestError(f"{where} does not exist: {candidate}")
    return candidate


def _audio_record(path: Path, where: str) -> dict[str, Any]:
    before = _identity(path)
    try:
        info = sf.info(path)
    except Exception as error:
        raise DynamicEndpointManifestError(f"{where} is not readable audio: {error}") from error
    if int(info.samplerate) != SAMPLE_RATE:
        raise DynamicEndpointManifestError(
            f"{where} must be {SAMPLE_RATE} Hz, got {info.samplerate}"
        )
    if int(info.channels) not in (1, 2) or int(info.frames) < 1:
        raise DynamicEndpointManifestError(
            f"{where} must be non-empty mono or stereo audio"
        )
    after = _identity(path)
    if before != after:
        raise DynamicEndpointManifestError(f"{where} changed while metadata was read")
    return {
        **before,
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": float(info.frames) / float(info.samplerate),
    }


def _structure_section_record(
    path: Path, *, label: str, occurrence: int
) -> dict[str, Any]:
    before = _identity(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise DynamicEndpointManifestError(
            f"target.structure_path is not valid JSON: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise DynamicEndpointManifestError("target structure must be a JSON object")

    rows = raw.get("sections")
    if not isinstance(rows, list):
        raise DynamicEndpointManifestError("target structure has no sections array")
    matching: list[tuple[float, float]] = []
    seen = 0
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("label")) != label:
            continue
        seen += 1
        row_occurrence = row.get("occurrence", seen)
        if row_occurrence == occurrence:
            start = row.get("start_sec", row.get("start"))
            stop = row.get("end_sec", row.get("end"))
            if isinstance(start, bool) or isinstance(stop, bool):
                raise DynamicEndpointManifestError("section bounds must be numeric")
            try:
                start_f, stop_f = float(start), float(stop)
            except (TypeError, ValueError) as error:
                raise DynamicEndpointManifestError("section bounds must be numeric") from error
            if not (math.isfinite(start_f) and math.isfinite(stop_f) and 0 <= start_f < stop_f):
                raise DynamicEndpointManifestError("section bounds are invalid")
            matching.append((start_f, stop_f))
    if len(matching) != 1:
        raise DynamicEndpointManifestError(
            f"section {label!r} occurrence {occurrence} was not uniquely found"
        )
    after = _identity(path)
    if before != after:
        raise DynamicEndpointManifestError("target structure changed while it was read")
    return {
        **before,
        "selected_section": {
            "label": label,
            "occurrence": occurrence,
            "start_sec": matching[0][0],
            "end_sec": matching[0][1],
        },
    }


def _selected_section_audio_record(
    path: Path, *, start_sec: float, end_sec: float
) -> dict[str, Any]:
    start_frame = int(round(start_sec * SAMPLE_RATE))
    stop_frame = int(round(end_sec * SAMPLE_RATE))
    frames = stop_frame - start_frame
    if start_frame < 0 or frames < 1:
        raise DynamicEndpointManifestError("selected target section has invalid frames")
    before = _identity(path)
    try:
        with sf.SoundFile(path) as handle:
            handle.seek(start_frame)
            audio = handle.read(frames, dtype="float32", always_2d=True)
    except Exception as error:
        raise DynamicEndpointManifestError(
            f"selected target section cannot be decoded: {error}"
        ) from error
    if audio.shape[0] != frames or audio.shape[1] not in (1, 2):
        raise DynamicEndpointManifestError("selected target section decode shape changed")
    if not np.isfinite(audio).all():
        raise DynamicEndpointManifestError("selected target section contains NaN or Inf")
    if audio.shape[1] == 1:
        stereo = np.repeat(audio, 2, axis=1)
        channel_policy = "mono duplicated to stereo"
    else:
        stereo = audio
        channel_policy = "stereo unchanged"
    stereo = np.ascontiguousarray(stereo, dtype=np.float32)
    after = _identity(path)
    if before != after:
        raise DynamicEndpointManifestError("target audio changed while section was decoded")
    return {
        "start_sample": start_frame,
        "stop_sample_exclusive": stop_frame,
        "frames": frames,
        "channels_after_policy": 2,
        "dtype": "float32",
        "channel_policy": channel_policy,
        "decoded_stereo_sha256": hashlib.sha256(stereo.tobytes()).hexdigest(),
        "peak_abs": float(np.max(np.abs(stereo))),
        "samples_over_1_0": int(np.sum(np.abs(stereo) > 1.0)),
    }


def _validate_endpoint(
    raw: Any, *, index: int, base_dir: Path, renderer_mode: str
) -> CanonicalEndpoint:
    if not isinstance(raw, Mapping):
        raise DynamicEndpointManifestError(f"endpoints[{index}] must be an object")
    kind = raw.get("kind")
    if kind not in ENDPOINT_KINDS:
        raise DynamicEndpointManifestError(
            f"endpoints[{index}].kind must be one of {ENDPOINT_KINDS}"
        )
    fields = (
        {"kind", "text", "effect_mode"}
        if kind == "natural_language"
        else {"kind", "audio_path", "effect_mode"}
    )
    row = _exact_object(raw, fields, f"endpoints[{index}]")
    effect_mode = row["effect_mode"]
    if effect_mode not in EFFECT_MODES:
        raise DynamicEndpointManifestError(
            f"endpoints[{index}].effect_mode must be explicit and one of {EFFECT_MODES}"
        )
    if renderer_mode != "eq_reverb" and effect_mode != renderer_mode:
        raise DynamicEndpointManifestError(
            f"endpoints[{index}] mode {effect_mode!r} is incompatible with renderer mode "
            f"{renderer_mode!r}"
        )
    endpoint_id = f"endpoint_{index + 1:03d}"
    if kind == "natural_language":
        return CanonicalEndpoint(
            endpoint_id=endpoint_id,
            kind=kind,
            effect_mode=effect_mode,
            user_description=_exact_text(row["text"], f"endpoints[{index}].text"),
            source_audio=None,
        )
    audio_path = _resolve_path(
        row["audio_path"], base_dir, f"endpoints[{index}].audio_path"
    )
    return CanonicalEndpoint(
        endpoint_id=endpoint_id,
        kind=kind,
        effect_mode=effect_mode,
        user_description=None,
        source_audio=_audio_record(audio_path, f"endpoints[{index}].audio_path"),
    )


def validate_manifest(
    raw: Any, *, base_dir: Path, source_manifest: Mapping[str, str]
) -> CanonicalManifest:
    top = _exact_object(
        raw,
        {
            "schema_version",
            "target",
            "automation_instruction",
            "renderer_mode",
            "endpoints",
        },
        "manifest",
    )
    if top["schema_version"] != SCHEMA_VERSION:
        raise DynamicEndpointManifestError(
            f"manifest.schema_version must be {SCHEMA_VERSION!r}"
        )
    renderer_mode = top["renderer_mode"]
    if renderer_mode not in EFFECT_MODES:
        raise DynamicEndpointManifestError(
            f"manifest.renderer_mode must be explicit and one of {EFFECT_MODES}"
        )
    target_raw = _exact_object(
        top["target"],
        {"raw_stem_path", "structure_path", "stem", "section", "occurrence"},
        "target",
    )
    stem = target_raw["stem"]
    if stem not in TARGET_STEMS:
        raise DynamicEndpointManifestError(f"target.stem must be one of {TARGET_STEMS}")
    section = target_raw["section"]
    if not isinstance(section, str) or not _SECTION_LABEL.fullmatch(section):
        raise DynamicEndpointManifestError("target.section must be a short plain label")
    if _INTERNAL_ENDPOINT_TOKEN.search(section):
        raise DynamicEndpointManifestError(
            "target.section cannot name a hidden internal endpoint identifier"
        )
    occurrence = _positive_int(target_raw["occurrence"], "target.occurrence")
    raw_stem_path = _resolve_path(
        target_raw["raw_stem_path"], base_dir, "target.raw_stem_path"
    )
    structure_path = _resolve_path(
        target_raw["structure_path"], base_dir, "target.structure_path"
    )
    target_audio = _audio_record(raw_stem_path, "target.raw_stem_path")
    structure = _structure_section_record(
        structure_path, label=section, occurrence=occurrence
    )
    selected = structure["selected_section"]
    if float(target_audio["duration_sec"]) + 1e-9 < float(selected["end_sec"]):
        raise DynamicEndpointManifestError(
            "target raw stem ends before the selected structure section"
        )
    endpoints_raw = top["endpoints"]
    if not isinstance(endpoints_raw, list) or not 1 <= len(endpoints_raw) <= MAX_USER_ENDPOINTS:
        raise DynamicEndpointManifestError(
            f"manifest.endpoints must contain 1..{MAX_USER_ENDPOINTS} entries"
        )
    endpoints = tuple(
        _validate_endpoint(
            endpoint,
            index=index,
            base_dir=base_dir,
            renderer_mode=renderer_mode,
        )
        for index, endpoint in enumerate(endpoints_raw)
    )
    target = {
        "raw_stem": target_audio,
        "structure": structure,
        "selected_section_audio": _selected_section_audio_record(
            raw_stem_path,
            start_sec=float(selected["start_sec"]),
            end_sec=float(selected["end_sec"]),
        ),
        "stem": stem,
        "section": section,
        "occurrence": occurrence,
    }
    return CanonicalManifest(
        source_manifest=dict(source_manifest),
        target=target,
        automation_instruction=_exact_text(
            top["automation_instruction"], "manifest.automation_instruction"
        ),
        renderer_mode=renderer_mode,
        endpoints=endpoints,
    )


def load_manifest(path: Path) -> CanonicalManifest:
    path = path.expanduser().resolve()
    identity = _identity(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise DynamicEndpointManifestError(f"manifest is not valid JSON: {error}") from error
    canonical = validate_manifest(raw, base_dir=path.parent, source_manifest=identity)
    if _identity(path) != identity:
        raise DynamicEndpointManifestError("manifest changed while it was loaded")
    return canonical
