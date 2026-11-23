"""Reference matching setup: checkpoint lookup and a peak-normalized encoder."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

import reference_bridge as bridge
import reference_matcher as matcher
from automation_fx import bezier, eq, reverb
from automation_fx.paths import FXENCODER_CACHE

ROOT = Path(__file__).resolve().parent
RUNTIME_CACHE_ROOT = ROOT / "demo_outputs" / ".runtime_cache" / "medleydb_reference_set5_v1"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_ROOT))


FXENCODER_CACHE_ROOT = Path(str(FXENCODER_CACHE))
FXENCODER_CHECKPOINT_NAME = "fxenc_plusplus_default.pt"


class ReferenceSetupError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locate_cached_checkpoint() -> Path:
    candidates = sorted(
        FXENCODER_CACHE_ROOT.glob(
            f"models--yytung--fxencoder-plusplus/snapshots/*/{FXENCODER_CHECKPOINT_NAME}"
        )
    )
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise ReferenceSetupError(
            f"expected one cached Fx-Encoder++ checkpoint, found {len(candidates)}"
        )
    return candidates[0]


def _verify_official_checkout() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(matcher.OFFICIAL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(matcher.OFFICIAL_REPO), "status", "--porcelain"], text=True
    ).strip()
    if head != matcher.OFFICIAL_COMMIT or dirty:
        raise ReferenceSetupError("pinned Fx-Encoder++ checkout changed")
    return {"path": str(matcher.OFFICIAL_REPO), "commit": head, "clean": True}




class ReferenceMatchError(RuntimeError):
    pass


class DifferentiablePeakNormalizedEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def get_fx_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor) or audio.ndim != 3:
            raise ReferenceMatchError(
                "encoder input must have shape [batch, channels, samples]"
            )
        if not audio.is_floating_point() or not bool(torch.isfinite(audio).all()):
            raise ReferenceMatchError(
                "encoder input must be finite floating-point audio"
            )
        peak = audio.abs().amax(dim=(-2, -1), keepdim=True)
        if bool((peak <= 0.0).any()):
            raise ReferenceMatchError("encoder input is silent")
        return self.encoder.get_fx_embedding(audio / peak)


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _reproducibility_identity() -> dict[str, Any]:
    checkpoint = _locate_cached_checkpoint()
    checkout = _verify_official_checkout()
    return {
        "code": {
            "runner": _identity(Path(__file__)),
            "runner_v1_support": _identity(Path(__file__)),
            "bridge": _identity(Path(bridge.__file__)),
            "bridge_base": _identity(Path(bridge.__file__)),
            "matcher": _identity(Path(matcher.__file__)),
            "eq_renderer": _identity(Path(eq.__file__)),
            "reverb_renderer": _identity(Path(reverb.__file__)),
            "numeric_plans": _identity(Path(bezier.__file__)),
            "eq_runtime": _identity(Path(eq.__file__)),
            "multiband_eq_runtime": _identity(Path(eq.__file__)),
            "torch_multiband_eq_runtime": _identity(
                Path(eq.__file__)
            ),
            "reverb_send_runtime": _identity(Path(reverb.__file__)),
        },
        "checkpoint": {
            **_identity(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "downloaded_during_prepare": False,
        },
        "official_checkout": {
            **checkout,
            "expected_commit": matcher.OFFICIAL_COMMIT,
        },
    }
