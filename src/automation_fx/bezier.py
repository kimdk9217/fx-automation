"""Cubic Bezier segments and their evaluation on a time grid."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

REGION_HZ = {
    "sub": (20.0, 60.0),
    "low": (60.0, 250.0),
    "mid": (250.0, 2000.0),
    "presence": (2000.0, 6000.0),
    "high": (6000.0, 12000.0),
    "air": (12000.0, None),
}


class EQSemanticPlanError(Exception):
    pass


class UnifiedPlanError(EQSemanticPlanError):
    pass


class NumericPlanError(UnifiedPlanError):
    pass


@dataclass(frozen=True)
class BezierSegment:
    p0: tuple
    p1: tuple
    p2: tuple
    p3: tuple

    def to_dict(self) -> dict:
        return {
            name: {"time": getattr(self, name)[0], "value": getattr(self, name)[1]}
            for name in ("p0", "p1", "p2", "p3")
        }


@dataclass(frozen=True)
class BezierEventTrajectory:
    segments: tuple
    end_behavior: str
    kind: str = "bezier_event"

    def to_dict(self) -> dict:
        return {
            "kind": "bezier_event",
            "end_behavior": self.end_behavior,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def peak_time(self) -> float:
        best = (0.0, 0.0)
        for segment in self.segments:
            for point in (segment.p0, segment.p1, segment.p2, segment.p3):
                if point[1] > best[1]:
                    best = point
        return best[0]


BISECTION_ITERATIONS = 60
BISECTION_TIME_TOLERANCE = 1e-12


class Curve2DError(NumericPlanError):
    pass


def segment_matrix(trajectory) -> np.ndarray:
    rows = []
    for segment in trajectory.segments:
        rows.append(
            [list(segment.p0), list(segment.p1), list(segment.p2), list(segment.p3)]
        )
    return np.array(rows, dtype=np.float64)


def control_points_sha256(trajectory) -> str:
    payload = trajectory.end_behavior + "|" + "|".join(
        f"{value:.17g}" for value in segment_matrix(trajectory).reshape(-1)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def curve_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def normalized_time_grid(n_samples: int) -> np.ndarray:
    if n_samples < 2:
        raise Curve2DError(f"automation span is too short: {n_samples} samples")
    grid = np.arange(n_samples, dtype=np.float64) / float(n_samples - 1)
    grid[-1] = 1.0
    return grid
