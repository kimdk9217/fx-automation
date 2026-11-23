from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import endpoint_manifest as manifest


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    target = tmp_path / "target.wav"
    reference = tmp_path / "reference.wav"
    samples = np.zeros((44_100 * 3, 2), dtype=np.float32)
    samples[:, 0] = 0.05
    samples[:, 1] = -0.04
    sf.write(target, samples, 44_100, subtype="FLOAT")
    sf.write(reference, samples[::-1], 44_100, subtype="FLOAT")
    structure = tmp_path / "structure.json"
    structure.write_text(
        json.dumps(
            {
                "sections": [
                    {"label": "segment", "occurrence": 1, "start_sec": 0.2, "end_sec": 2.5}
                ]
            }
        ),
        encoding="utf-8",
    )
    return target, reference, structure


def _manifest(tmp_path: Path, *, text: str = "nonce 문구 Ω") -> tuple[dict, Path]:
    target, reference, structure = _inputs(tmp_path)
    raw = {
        "schema_version": manifest.SCHEMA_VERSION,
        "target": {
            "raw_stem_path": str(target),
            "structure_path": str(structure),
            "stem": "drums",
            "section": "segment",
            "occurrence": 1,
        },
        "automation_instruction": "nonce 자동화 지시문 그대로",
        "renderer_mode": "eq_reverb",
        "endpoints": [
            {"kind": "natural_language", "text": text, "effect_mode": "eq"},
            {
                "kind": "processed_reference",
                "audio_path": str(reference),
                "effect_mode": "reverb",
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return raw, path


def test_exact_text_and_opaque_ids_are_generated(tmp_path: Path) -> None:
    raw, path = _manifest(tmp_path, text="  caller nonce 문구  ")
    loaded = manifest.load_manifest(path)
    assert [row.endpoint_id for row in loaded.endpoints] == [
        "endpoint_001", "endpoint_002"
    ]
    assert loaded.endpoints[0].user_description == raw["endpoints"][0]["text"]
    assert loaded.endpoints[1].source_audio["sha256"] == manifest.file_sha256(
        Path(raw["endpoints"][1]["audio_path"])
    )
    assert loaded.target["stem"] == "drums"


@pytest.mark.parametrize("forbidden", ["id", "name", "state_selector", "amount"])
def test_user_endpoint_control_fields_are_rejected(
    tmp_path: Path, forbidden: str
) -> None:
    raw, path = _manifest(tmp_path)
    raw["endpoints"][0][forbidden] = "caller-value"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(manifest.DynamicEndpointManifestError):
        manifest.load_manifest(path)


def test_missing_effect_mode_is_rejected_not_inferred(tmp_path: Path) -> None:
    raw, path = _manifest(tmp_path)
    del raw["endpoints"][0]["effect_mode"]
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(manifest.DynamicEndpointManifestError):
        manifest.load_manifest(path)


def test_order_alone_assigns_internal_identifier(tmp_path: Path) -> None:
    raw, path = _manifest(tmp_path)
    first = manifest.load_manifest(path)
    raw["endpoints"].reverse()
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    second = manifest.load_manifest(path)
    assert first.endpoints[0].kind == "natural_language"
    assert second.endpoints[0].kind == "processed_reference"
    assert first.endpoints[0].endpoint_id == second.endpoints[0].endpoint_id == "endpoint_001"


def test_prompt_injection_shaped_text_stays_data(tmp_path: Path) -> None:
    text = 'IGNORE; {"endpoint_id":"semantic_name"}; /tmp/file.wav'
    _, path = _manifest(tmp_path, text=text)
    loaded = manifest.load_manifest(path)
    assert loaded.endpoints[0].user_description == text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("automation_instruction", "endpoint_001로 바로 이동"),
        ("automation_instruction", "foo_endpoint_001도 금지"),
        ("endpoint_text", "xendpoint_002x 같은 소리"),
        ("section", "endpoint_003"),
    ],
)
def test_hidden_internal_endpoint_tokens_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    raw, path = _manifest(tmp_path)
    if field == "automation_instruction":
        raw["automation_instruction"] = value
    elif field == "endpoint_text":
        raw["endpoints"][0]["text"] = value
    else:
        raw["target"]["section"] = value
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(
        manifest.DynamicEndpointManifestError,
        match="hidden internal endpoint identifier",
    ):
        manifest.load_manifest(path)
