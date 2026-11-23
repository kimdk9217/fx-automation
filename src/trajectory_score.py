"""Rank correlation and permutation test for direction scoring."""

from __future__ import annotations

import numpy as np


def _rank(values):
    return np.argsort(np.argsort(np.asarray(values, dtype=np.float64))).astype(np.float64)


def rank_corr(a, b):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else None


def permutation_test(d_source, d_target, n_permutations: int = 2000,
                     seed: int = 0) -> dict:
    diff = np.asarray(d_source, dtype=np.float64) - np.asarray(d_target, dtype=np.float64)
    observed = rank_corr(np.arange(diff.size), diff)
    if observed is None:
        return {"score": None, "p_value": None, "n_permutations": 0,
                "why": "all values are equal, so rank correlation is undefined"}
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    index = np.arange(diff.size)
    for i in range(n_permutations):
        null[i] = rank_corr(index, rng.permutation(diff))

    greater = int(np.sum(null >= observed))
    return {"score": float(observed),
            "p_value": float((greater + 1) / (n_permutations + 1)),
            "null_median": float(np.median(null)),
            "null_p95": float(np.percentile(null, 95)),
            "n_permutations": int(n_permutations)}


def score_run(distances: dict, source: str, target: str, n_permutations: int = 2000,
              seed: int = 0) -> dict:
    src = np.asarray(distances[source], dtype=np.float64)
    tgt = np.asarray(distances[target], dtype=np.float64)
    if src.ndim == 1:
        src, tgt = src[None, :], tgt[None, :]
    per_track = [permutation_test(src[i], tgt[i], n_permutations, seed + i)
                 for i in range(src.shape[0])]
    scores = [r["score"] for r in per_track if r["score"] is not None]
    return {
        "source": source, "target": target,
        "per_track": per_track,
        "n_tracks": src.shape[0],
        "score_median": float(np.median(scores)) if scores else None,
        "tracks_significant": sum(1 for r in per_track
                                  if r["p_value"] is not None and r["p_value"] <= 0.05),
        "pooled": permutation_test(src.mean(axis=0), tgt.mean(axis=0),
                                   n_permutations, seed),
    }
