"""Peak level report."""

from __future__ import annotations

import numpy as np


def peak_report(data: np.ndarray) -> dict:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    over = int(np.count_nonzero(np.abs(data) > 1.0))
    return {
        "peak_abs": peak,
        "peak_dbfs": float(20.0 * np.log10(peak)) if peak > 0.0 else float("-inf"),
        "samples_over_1.0": over,
        "clipping_risk": peak > 1.0,
    }
