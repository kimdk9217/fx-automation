"""Signed EQ automation curve: validate planner events, build the Bezier curve and evaluate it."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from automation_fx.bezier import (
    BISECTION_ITERATIONS,
    BISECTION_TIME_TOLERANCE,
    BezierEventTrajectory,
    BezierSegment,
    control_points_sha256,
    curve_sha256,
    normalized_time_grid,
    segment_matrix,
)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def softplus(value: float) -> float:
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _monotone_handles(
    start: float,
    end: float,
    first_coordinate: float,
    second_coordinate: float,
) -> tuple[float, float]:
    first = start + (end - start) * sigmoid(first_coordinate)
    second = first + (end - first) * sigmoid(second_coordinate)
    return first, second


def _cubic(c0, c1, c2, c3, u):
    a0 = c0 + u * (c1 - c0)
    a1 = c1 + u * (c2 - c1)
    a2 = c2 + u * (c3 - c2)
    b0 = a0 + u * (a1 - a0)
    b1 = a1 + u * (a2 - a1)
    return b0 + u * (b1 - b0)


def evaluate_signed_curve(trajectory: BezierEventTrajectory, n_samples: int) -> dict[str, Any]:
    points = segment_matrix(trajectory)
    grid = normalized_time_grid(n_samples)
    starts = points[:, 0, 0]
    indices = np.maximum(np.searchsorted(starts, grid, side="right") - 1, 0)
    values = np.empty(n_samples, dtype=np.float64)
    residual = np.zeros(n_samples, dtype=np.float64)
    for segment_index in range(points.shape[0]):
        mask = indices == segment_index
        if not np.any(mask):
            continue
        x0, x1, x2, x3 = points[segment_index, :, 0]
        y0, y1, y2, y3 = points[segment_index, :, 1]
        target = grid[mask]
        lo = np.zeros(target.shape, dtype=np.float64)
        hi = np.ones(target.shape, dtype=np.float64)
        for _ in range(BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            below = _cubic(x0, x1, x2, x3, mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        x_lo = _cubic(x0, x1, x2, x3, lo)
        x_hi = _cubic(x0, x1, x2, x3, hi)
        u = np.where(np.abs(x_lo - target) <= np.abs(x_hi - target), lo, hi)
        u = np.where(target == x0, 0.0, u)
        u = np.where((target == x3) & (target != x0), 1.0, u)
        values[mask] = _cubic(y0, y1, y2, y3, u)
        residual[mask] = np.abs(_cubic(x0, x1, x2, x3, u) - target)

    boundary_gaps = [
        abs(float(points[i, 3, 1]) - float(points[i + 1, 0, 1]))
        for i in range(points.shape[0] - 1)
    ]
    metrics = {
        "n_samples": int(n_samples),
        "n_segments": int(points.shape[0]),
        "start_value": float(values[0]),
        "end_value": float(values[-1]),
        "min_value": float(np.min(values)),
        "max_value": float(np.max(values)),
        "max_time_residual": float(np.max(residual)),
        "residual_within_tolerance": bool(
            np.max(residual) <= BISECTION_TIME_TOLERANCE
        ),
        "max_segment_boundary_gap": max(boundary_gaps, default=0.0),
        "curve_sha256": curve_sha256(values),
        "control_points_sha256": control_points_sha256(trajectory),
        "value_range": [-1.0, 1.0],
        "values_clamped": False,
        "control_points_sorted": False,
        "category_template_used": False,
    }
    if values[0] != points[0, 0, 1] or values[-1] != points[-1, 3, 1]:
        raise ValueError("signed curve endpoint was not represented exactly")
    if metrics["min_value"] < -1.0 or metrics["max_value"] > 1.0:
        raise ValueError("signed curve left [-1,1]")
    return {"values": values, "time": grid, "metrics": metrics}


RELATIONS = ("set_state", "hold_current")


FIELDS = (
    "id",
    "instruction_evidence",
    "relation",
    "level",
    "duration_coordinate",
    "handle1_time_coordinate",
    "handle2_time_coordinate",
    "handle1_value_coordinate",
    "handle2_value_coordinate",
)


def _finite(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _check_event_rows(raw: Any, *, instruction: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"kind", "events"}:
        raise ValueError("trajectory must contain exactly kind and events")
    if raw["kind"] != "signed_level_event_bezier_v3":
        raise ValueError("trajectory.kind must be signed_level_event_bezier_v3")
    rows = raw["events"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        raise ValueError("trajectory.events must contain 1 to 6 events")
    clean = []
    seen = set()
    evidence_cursor = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError(f"events[{index}] fields do not match contract")
        event_id = row["id"]
        if not isinstance(event_id, str) or not event_id or len(event_id) > 32:
            raise ValueError(f"events[{index}].id is invalid")
        if event_id in seen:
            raise ValueError(f"duplicate event id: {event_id}")
        evidence = row["instruction_evidence"]
        if not isinstance(evidence, str) or not evidence:
            raise ValueError(f"events[{index}].instruction_evidence is invalid")
        if instruction is not None:
            position = instruction.find(evidence, evidence_cursor)
            if position < 0:
                raise ValueError(f"events[{index}].instruction_evidence is not ordered evidence")
            evidence_cursor = position + len(evidence)
        relation = row["relation"]
        if relation not in RELATIONS:
            raise ValueError(f"events[{index}].relation is invalid")
        if relation == "set_state":
            level = _finite(row["level"], f"events[{index}].level")
            if not -1.0 <= level <= 1.0:
                raise ValueError(f"events[{index}].level must be inside [-1,1]")
        else:
            if row["level"] is not None:
                raise ValueError(f"events[{index}].level must be null for {relation}")
            level = None
        clean.append(
            {
                "id": event_id,
                "instruction_evidence": evidence,
                "relation": relation,
                "level": level,
                "duration_coordinate": _finite(
                    row["duration_coordinate"], f"events[{index}].duration_coordinate"
                ),
                "handle1_time_coordinate": _finite(
                    row["handle1_time_coordinate"], f"events[{index}].handle1_time_coordinate"
                ),
                "handle2_time_coordinate": _finite(
                    row["handle2_time_coordinate"], f"events[{index}].handle2_time_coordinate"
                ),
                "handle1_value_coordinate": _finite(
                    row["handle1_value_coordinate"], f"events[{index}].handle1_value_coordinate"
                ),
                "handle2_value_coordinate": _finite(
                    row["handle2_value_coordinate"], f"events[{index}].handle2_value_coordinate"
                ),
            }
        )
        seen.add(event_id)
    return {"kind": raw["kind"], "events": clean}


def _check_initial_level(raw: Any, *, instruction: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"kind", "initial_level", "events"}:
        raise ValueError("trajectory must contain exactly kind, initial_level and events")
    if raw["kind"] != "signed_level_event_bezier_v4":
        raise ValueError("trajectory.kind must be signed_level_event_bezier_v4")
    initial = raw["initial_level"]
    if isinstance(initial, bool) or not isinstance(initial, (int, float)):
        raise ValueError("trajectory.initial_level must be a finite number")
    initial = float(initial)
    if not np.isfinite(initial) or not -1.0 <= initial <= 1.0:
        raise ValueError("trajectory.initial_level must be inside [-1,1]")

    rows = raw["events"]
    if not isinstance(rows, list) or len(rows) > 6:
        raise ValueError("trajectory.events must contain 0 to 6 events")
    if rows:
        checked = _check_event_rows(
            {"kind": "signed_level_event_bezier_v3", "events": rows},
            instruction=instruction)
        clean = checked["events"]
    else:
        clean = []
    return {"kind": raw["kind"], "initial_level": initial, "events": clean}


def _build_bezier(raw: Any, *, instruction: str | None = None):
    trajectory = _check_initial_level(raw, instruction=instruction)
    rows = trajectory["events"]
    initial = trajectory["initial_level"]

    if not rows:
        points = ((0.0, initial), (1 / 3, initial), (2 / 3, initial), (1.0, initial))
        segment = BezierSegment(*points)
        typed = BezierEventTrajectory((segment,), "signed_level_v4")
        return typed, {
            "resolved_control_points": {"kind": "signed_level_bezier_event_v4",
                                        "segments": [segment.to_dict()]},
            "initial_level": initial,
            "events": [],
            "normalized_duration_fractions": np.asarray([1.0]),
            "value_range": [-1.0, 1.0],
            "level_owner": "LLM signed value used exactly",
            "level_transform": "none",
            "held_constant_at_initial_level": True,
        }

    weights = np.asarray(
        [softplus(row["duration_coordinate"]) + 1e-12 for row in rows],
        dtype=np.float64)
    fractions = weights / float(np.sum(weights))
    boundaries = np.concatenate(([0.0], np.cumsum(fractions)))
    boundaries[-1] = 1.0

    previous = initial
    segments, control_points, report_rows = [], [], []
    for index, row in enumerate(rows):
        relation = row["relation"]
        endpoint = float(row["level"]) if relation == "set_state" else previous
        start_time = float(boundaries[index])
        end_time = float(boundaries[index + 1])
        first_time = start_time + (end_time - start_time) * sigmoid(
            row["handle1_time_coordinate"])
        second_time = first_time + (end_time - first_time) * sigmoid(
            row["handle2_time_coordinate"])
        if relation == "hold_current":
            first_value = second_value = previous
        else:
            first_value, second_value = _monotone_handles(
                previous, endpoint,
                row["handle1_value_coordinate"], row["handle2_value_coordinate"])
        points = ((start_time, previous), (first_time, first_value),
                  (second_time, second_value), (end_time, endpoint))
        segment = BezierSegment(*points)
        segments.append(segment)
        control_points.append(segment.to_dict())
        report_rows.append({**row, "start_level": previous,
                            "resolved_end_level": endpoint,
                            "normalized_duration_fraction": float(fractions[index])})
        previous = endpoint

    typed = BezierEventTrajectory(tuple(segments), "signed_level_v4")
    return typed, {
        "resolved_control_points": {"kind": "signed_level_bezier_event_v4",
                                    "segments": control_points},
        "initial_level": initial,
        "events": report_rows,
        "normalized_duration_fractions": fractions,
        "value_range": [-1.0, 1.0],
        "level_owner": "LLM signed value used exactly",
        "level_transform": "none",
        "fixed_curve_template": False,
        "style_specific_rules": False,
        "sorted": False,
        "clamped": False,
        "repaired": False,
        "starts_at_initial_level_not_zero": bool(initial != 0.0),
    }


def _occurrences(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        position = haystack.find(needle, start)
        if position < 0:
            return out
        out.append(position)
        start = position + 1


def _greedy_forward_positions(instruction: str, evidences: Sequence[str]):
    cursor, chosen = 0, []
    for evidence in evidences:
        position = instruction.find(evidence, cursor)
        if position < 0:
            return None
        chosen.append(position)
        cursor = position + len(evidence)
    return chosen


def ground_evidence(evidences: Sequence[str], instruction: str,
                    prior_instructions: Sequence[str] = ()) -> dict:
    sources = [("current", instruction)] + [
        (f"turn-{i}", text) for i, text in enumerate(prior_instructions)]
    rows, missing = [], []
    for index, evidence in enumerate(evidences):
        found = None
        for name, text in sources:
            positions = _occurrences(text, evidence)
            if positions:
                found = {"event_index": index, "source": name,
                         "positions": positions, "first_position": positions[0]}
                break
        if found is None:
            missing.append(index)
        else:
            rows.append(found)
    ordered = _greedy_forward_positions(instruction, evidences)
    from collections import Counter
    repeats = {text: count for text, count in Counter(evidences).items() if count > 1}
    return {
        "grounded": rows,
        "missing_event_indexes": missing,
        "text_ordered": ordered is not None,
        "forward_positions": ordered,
        "reused_evidence": repeats,
        "v4_would_accept": bool(not missing and ordered is not None),
        "rule": ("Evidence must appear in the instruction. Whether text order matches "
                 "time order is measured and recorded, not enforced."),
    }


def validate_events(raw: Any, *, instruction: str | None = None,
                    prior_instructions: Sequence[str] = ()) -> dict[str, Any]:
    checked = _check_initial_level(raw, instruction=None)
    evidences = [row["instruction_evidence"] for row in checked["events"]]
    if instruction is None:
        checked["evidence_report"] = {"checked": False,
                                      "why": "no instruction was given, so evidence was not checked"}
        return checked
    report = ground_evidence(evidences, instruction, prior_instructions)
    if report["missing_event_indexes"]:
        first = report["missing_event_indexes"][0]
        raise ValueError(
            f"events[{first}].instruction_evidence is not grounded in the instruction")
    checked["evidence_report"] = {"checked": True, **report}
    return checked


def resolve_signed_curve(raw: Any, *, instruction: str | None = None,
                         prior_instructions: Sequence[str] = ()):
    trajectory = validate_events(raw, instruction=instruction,
                                 prior_instructions=prior_instructions)
    typed, resolved = _build_bezier(
        {"kind": trajectory["kind"], "initial_level": trajectory["initial_level"],
         "events": [{k: v for k, v in row.items()} for row in trajectory["events"]]},
        instruction=None)
    resolved["evidence_report"] = trajectory["evidence_report"]
    return typed, resolved
