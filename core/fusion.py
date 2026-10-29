"""🧪 Fusion de listes de résultats par Reciprocal Rank Fusion (PLAN.md phase 4.2).

`rrf(lists, k=60, weights=None)` : chaque liste est une suite de (id, score) triée
par pertinence décroissante ; le score fusionné d'un id est Σ_l w_l / (k + rang_l(id)),
rang à partir de 1. Les ids absents d'une liste n'y contribuent pas. `k` ∈ {10, 30, 60}
est comparé dans l'évaluation ; la variante pondérée sert à privilégier une source.

`soft_bonus(hits, meta, hints)` : filtres doux de la phase 5 — bonus multiplicatif
sur la décennie et la langue, jamais une exclusion.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

Hit = tuple[str, float]


def rrf(lists: Sequence[Iterable[Hit]], k: float = 60.0, weights: Sequence[float] | None = None,
        top: int | None = None) -> list[Hit]:
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights doit avoir la longueur de lists")
    scores: dict[str, float] = defaultdict(float)
    for w, lst in zip(weights, lists):
        if not w:
            continue
        seen: set[str] = set()
        rank = 0
        for item in lst:
            id_ = item[0]
            if id_ in seen:
                continue
            seen.add(id_)
            rank += 1
            scores[id_] += w / (k + rank)
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return out[:top] if top else out


def normalize_scores(hits: Iterable[Hit]) -> list[Hit]:
    """Min-max sur une liste (pour la fusion par scores, baseline de comparaison)."""
    hits = list(hits)
    if not hits:
        return []
    vals = [s for _, s in hits]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [(i, 1.0) for i, _ in hits]
    return [(i, (s - lo) / (hi - lo)) for i, s in hits]


def soft_bonus(hits: Iterable[Hit], meta: Mapping[str, Mapping], hints: Mapping[str, object],
               decade_bonus: float = 0.15, language_bonus: float = 0.15, neighbor_decade_bonus: float = 0.05) -> list[Hit]:
    """Bonus multiplicatif si la décennie / la langue de la fiche correspond aux indices.
    `hints` : {"decade": "1980s", "language": "french"} (valeurs absentes ignorées)."""
    dec = hints.get("decade")
    lang = hints.get("language")
    dec_val = int(str(dec)[:4]) if dec and str(dec)[:4].isdigit() else None
    out: list[Hit] = []
    for id_, s in hits:
        m = meta.get(id_, {})
        f = 1.0
        y = m.get("year")
        if dec_val is not None and y:
            d = int(y) // 10 * 10
            if d == dec_val:
                f += decade_bonus
            elif abs(d - dec_val) == 10:
                f += neighbor_decade_bonus
        if lang and m.get("language") and str(m["language"]).lower() == str(lang).lower():
            f += language_bonus
        out.append((id_, s * f))
    out.sort(key=lambda x: -x[1])
    return out
