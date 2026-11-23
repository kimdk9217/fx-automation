"""Music structure analysis with All-In-One: beats, downbeats and labelled sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SCHEMA_VERSION = "structure-1"


class StructureError(Exception):
    pass


class StructureBackendError(StructureError):
    pass


@dataclass(frozen=True)
class Section:
    label: str
    start_sec: float
    end_sec: float
    occurrence: int
    segment_id: int = -1

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Structure:
    bpm: float
    beats: list[float]
    downbeats: list[float]
    sections: list[Section]
    source: str = "unknown"
    audio: dict = field(default_factory=dict)
    backend: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(s.segment_id < 0 for s in self.sections):
            object.__setattr__(
                self,
                "sections",
                [s if s.segment_id >= 0 else replace(s, segment_id=i)
                 for i, s in enumerate(self.sections)],
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "bpm": self.bpm,
            "beats": list(self.beats),
            "downbeats": list(self.downbeats),
            "sections": [asdict(s) for s in self.sections],
            "source": self.source,
            "audio": dict(self.audio),
            "backend": dict(self.backend),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Structure":
        version = data.get("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            raise StructureError(
                f"structure schema version differs: {version!r} != {SCHEMA_VERSION!r}"
            )
        required = ("bpm", "beats", "downbeats", "sections")
        missing = [k for k in required if k not in data]
        if missing:
            raise StructureError(f"structure is missing required fields: {missing}")
        sections = [
            Section(
                label=str(s["label"]),
                start_sec=float(s["start_sec"]),
                end_sec=float(s["end_sec"]),
                occurrence=int(s["occurrence"]),
                segment_id=int(s.get("segment_id", i)),
            )
            for i, s in enumerate(data["sections"])
        ]
        return cls(
            bpm=float(data["bpm"]),
            beats=[float(b) for b in data["beats"]],
            downbeats=[float(b) for b in data["downbeats"]],
            sections=sections,
            source=str(data.get("source", "unknown")),
            audio=dict(data.get("audio", {})),
            backend=dict(data.get("backend", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "Structure":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def labels(self) -> list[str]:
        seen: list[str] = []
        for s in self.sections:
            if s.label not in seen:
                seen.append(s.label)
        return seen

    def summary(self) -> dict:
        return {
            "bpm": self.bpm,
            "n_beats": len(self.beats),
            "n_downbeats": len(self.downbeats),
            "available_sections": [
                {
                    "label": s.label,
                    "occurrence": s.occurrence,
                    "start_sec": round(s.start_sec, 3),
                    "end_sec": round(s.end_sec, 3),
                }
                for s in self.sections
            ],
        }


def find_section(structure: Structure, label: str, occurrence: int = 1) -> Section:
    if occurrence < 1:
        raise StructureError(f"occurrence must be >= 1: {occurrence}")
    matches = [s for s in structure.sections if s.label == label]
    if not matches:
        raise StructureError(
            f"section label {label!r} is not in the structure; available: {structure.labels()}"
        )
    if occurrence > len(matches):
        raise StructureError(
            f"section {label!r} occurrence {occurrence} is out of range ({len(matches)} found)"
        )
    return matches[occurrence - 1]


def audio_fingerprint(audio_path: str | Path) -> dict:
    path = Path(audio_path)
    if not path.is_file():
        raise StructureError(f"audio file not found: {path}")
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def analyze_with_all_in_one(
    audio_path: str | Path,
    device: str = "cpu",
    model: str = "harmonix-all",
    work_dir: str | Path | None = None,
    keep_byproducts: bool = False,
) -> Structure:
    try:
        import allin1
    except ImportError as exc:
        raise StructureBackendError(
            "All-In-One is not installed.\n"
            "  pip install allin1\n"
            "It also needs PyTorch and NATTEN, and downloads the Demucs/All-In-One "
            "checkpoints on first run."
        ) from exc

    import time

    work_dir = Path(work_dir) if work_dir is not None else Path(audio_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = allin1.analyze(
        str(audio_path),
        device=device,
        model=model,
        demix_dir=str(work_dir / "demix"),
        spec_dir=str(work_dir / "spec"),
        keep_byproducts=keep_byproducts,
    )
    elapsed = time.time() - started

    structure = structure_from_all_in_one(result, audio_path)
    return Structure(
        bpm=structure.bpm,
        beats=structure.beats,
        downbeats=structure.downbeats,
        sections=structure.sections,
        source=structure.source,
        audio=structure.audio,
        backend={
            "package": "allin1",
            "package_version": _package_version("allin1"),
            "model": model,
            "device": device,
            "torch_version": _package_version("torch"),
            "natten_version": _package_version("natten"),
            "demucs_version": _package_version("demucs"),
            "madmom_version": _package_version("madmom"),
            "elapsed_sec": round(elapsed, 3),
            "result_path": str(getattr(result, "path", audio_path)),
            "is_real_model_output": True,
        },
    )


def structure_from_all_in_one(result, audio_path: str | Path) -> Structure:
    missing = [name for name in ("bpm", "beats", "downbeats", "segments") if not hasattr(result, name)]
    if missing:
        raise StructureBackendError(
            f"All-In-One result is missing expected fields: {missing}"
        )

    counts: dict[str, int] = {}
    sections: list[Section] = []
    for segment in result.segments:
        for name in ("label", "start", "end"):
            if not hasattr(segment, name):
                raise StructureBackendError(
                    f"All-In-One segment has no {name} field"
                )
        label = str(segment.label)
        counts[label] = counts.get(label, 0) + 1
        sections.append(
            Section(label=label, start_sec=float(segment.start), end_sec=float(segment.end),
                    occurrence=counts[label], segment_id=len(sections))
        )

    return Structure(
        bpm=float(result.bpm),
        beats=[float(b) for b in result.beats],
        downbeats=[float(b) for b in result.downbeats],
        sections=sections,
        source="all_in_one",
        audio=audio_fingerprint(audio_path),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(analyze_with_all_in_one(args.audio, device=args.device).save_json(args.output))
