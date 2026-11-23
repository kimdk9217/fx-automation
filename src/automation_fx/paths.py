"""Paths to the repository, data archive and Fx-Encoder++ checkout. Override with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "src"
MEDLEYDB_ARCHIVE = Path(os.environ.get("MEDLEYDB_ARCHIVE", REPO_ROOT / "data/MedleyDB_V1.tar.gz"))
FXENCODER_REPO = Path(os.environ.get("FXENCODER_OFFICIAL_REPO", REPO_ROOT / "vendor/Fx-Encoder_PlusPlus"))
FXENCODER_CACHE = Path(os.environ.get("FXENCODER_CACHE", REPO_ROOT / ".cache/fxencoder_plusplus"))
