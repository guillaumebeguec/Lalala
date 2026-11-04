"""Retrieval metrics: top-k and MRR over lists of (id, score)."""

from __future__ import annotations

from typing import Iterable, Sequence


def rank_of(hits: Sequence[tuple[str, float]], answer: str) -> int | None:
    for i, (id_, _) in enumerate(hits, 1):
        if id_ == answer:
            return i
    return None


def summarize(ranks: Iterable[int | None], ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    ranks = list(ranks)
    n = len(ranks)
    out = {f"top{k}": (sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0) for k in ks}
    out["mrr"] = (sum(1.0 / r for r in ranks if r is not None) / n) if n else 0.0
    out["n"] = n
    return out


def bootstrap_ci(ranks: Sequence[int | None], k: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    import numpy as np
    rng = np.random.default_rng(seed)
    hits = np.array([1.0 if r is not None and r <= k else 0.0 for r in ranks])
    if len(hits) == 0:
        return 0.0, 0.0
    means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
