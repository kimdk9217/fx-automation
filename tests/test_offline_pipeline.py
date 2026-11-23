import importlib.util
import subprocess
from pathlib import Path

import pytest

import reference_matcher

ROOT = Path(__file__).resolve().parents[1]


def _pinned_checkout() -> bool:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(reference_matcher.OFFICIAL_REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return head == reference_matcher.OFFICIAL_COMMIT


@pytest.mark.skipif(not _pinned_checkout(), reason="needs the pinned Fx-Encoder++ submodule")
def test_offline_pipeline_runs_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location("offline_pipeline", ROOT / "examples" / "offline_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(tmp_path / "out")
    assert result["llm_calls"] == 2
    assert Path(result["rendered"]).is_file()
    assert (tmp_path / "out" / "evaluation.json").is_file()
