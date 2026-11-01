"""🧪 Reranking par Llama (cross-encodeur LLM) — PLAN.md phase 6.3.

Les embeddings sont des bi-encodeurs (requête et morceau encodés séparément). Ici
Llama voit **la description et la fiche ensemble** et note chaque candidat de 0 à 10
avec une justification courte. Lots de 5 candidats pour tenir dans le contexte du 8B.
Score final = `alpha · score_fusion_normalisé + (1 − alpha) · score_llama / 10`
(`alpha` réglé sur un sous-ensemble de T1). Cache par hash (description + fiche).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from core import llm
from core.cache import DiskCache
from core.config import get_settings
from core.fusion import normalize_scores

logger = logging.getLogger(__name__)

SCHEMA = {
    "title": "rerank",
    "type": "object",
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "score", "reason"],
                "properties": {"id": {"type": "string"}, "score": {"type": "number", "minimum": 0, "maximum": 10},
                               "reason": {"type": "string"}},
            },
        }
    },
}

_SYSTEM = (
    "You are a music expert judging whether a candidate song matches a vague description written by someone "
    "trying to remember a song. For EACH candidate, give a score from 0 (impossible) to 10 (almost certainly it) "
    "and a reason of at most 10 words. Judge on era, language, genre, sound, vocals, themes, usages (ads, films) "
    "and any quoted lyrics. `sound_terms` in the clues describe the instrumentation or voice the user remembers, NOT "
    "words to find in a title or in lyrics: a song titled 'Guitar' is no match for someone who remembers a guitar sound. "
    "Unknown facts are neutral, contradictions are penalised. JSON only: "
    "{\"scores\": [{\"id\": \"...\", \"score\": 0-10, \"reason\": \"...\"}, ...]} with exactly one entry per candidate id."
)


@dataclass
class Reranked:
    id: str
    fused: float
    llm_score: float | None
    reason: str
    final: float


def _score_batch(description: str, batch: Sequence[tuple[str, str]], model: str | None) -> dict[str, tuple[float, str]]:
    cache = DiskCache("rerank")
    key = cache.key(description + "\n" + "\n".join(f"{i}\t{c}" for i, c in batch), model=model or get_settings().llm_model, v=2)
    hit = cache.get(key)
    if hit is not None:
        return {k: tuple(v) for k, v in hit.items()}
    user = "Description:\n" + description.strip() + "\n\nCandidates:\n" + "\n\n".join(
        f"[id: {cid}]\n{card}" for cid, card in batch)
    out: dict[str, tuple[float, str]] = {}
    try:
        r = llm.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                     json_schema=SCHEMA, temperature=0.0, max_tokens=320, model=model)
        for item in (r.json or {}).get("scores", []):
            cid = str(item.get("id", "")).strip()
            try:
                sc = max(0.0, min(10.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                continue
            if cid in dict(batch):
                out[cid] = (sc, str(item.get("reason", ""))[:200])
    except llm.LLMError as e:
        logger.warning("rerank : %s", e)
    cache.put(key, {k: list(v) for k, v in out.items()})
    return out


LISTWISE_SCHEMA = {
    "title": "rerank_listwise",
    "type": "object",
    "required": ["ranking"],
    "properties": {"ranking": {"type": "array", "items": {"type": "object", "required": ["id"],
                                                          "properties": {"id": {"type": "string"}, "reason": {"type": "string"}}}}},
}

_LISTWISE_SYSTEM = (
    "You are a music expert. Someone describes a song they half-remember. You get N candidate songs with short cards. "
    "Return the candidates ORDERED from most to least likely (all of them), as JSON: "
    "{\"ranking\": [{\"id\": \"...\", \"reason\": \"<10 words\"}, ...]}. Judge era, language, genre, sound, themes, usages, lyrics."
)


def _rank_listwise(description: str, items: Sequence[tuple[str, str]], model: str | None) -> dict[str, tuple[float, str]]:
    """Mode listwise (recommandation 7, option 1) : un seul appel classe tous les candidats ;
    réservé aux modèles plus gros (contexte et qualité). Score = position inversée dans [0, 10]."""
    cache = DiskCache("rerank_listwise")
    key = cache.key(description + "\n" + "\n".join(f"{i}\t{c}" for i, c in items), model=model or get_settings().llm_model, v=2)
    hit = cache.get(key)
    if hit is not None:
        return {k: tuple(v) for k, v in hit.items()}
    user = "Description:\n" + description.strip() + f"\n\n{len(items)} candidates:\n" + "\n\n".join(f"[id: {cid}]\n{card}" for cid, card in items)
    out: dict[str, tuple[float, str]] = {}
    try:
        r = llm.chat([{"role": "system", "content": _LISTWISE_SYSTEM}, {"role": "user", "content": user}],
                     json_schema=LISTWISE_SCHEMA, temperature=0.0, max_tokens=900, model=model)
        valid = {cid for cid, _ in items}
        order = [str(x.get("id", "")).strip() for x in (r.json or {}).get("ranking", []) if isinstance(x, dict)]
        reasons = {str(x.get("id", "")).strip(): str(x.get("reason", ""))[:200] for x in (r.json or {}).get("ranking", []) if isinstance(x, dict)}
        order = [i for i in order if i in valid]
        n = len(valid)
        for pos, cid in enumerate(order):
            out[cid] = (10.0 * (n - pos) / n, reasons.get(cid, ""))
    except llm.LLMError as e:
        logger.warning("rerank listwise : %s", e)
    cache.put(key, {k: list(v) for k, v in out.items()})
    return out


def rerank(description: str, fused: Sequence[tuple[str, float]], cards: Mapping[str, str], alpha: float = 0.5,
           batch_size: int = 5, top: int = 20, model: str | None = None, mode: str | None = None) -> list[Reranked]:
    """`fused` : (track_id, score de fusion) triés ; `cards` : track_id → fiche courte.
    `mode` : `pointwise` (lots de 5 notés 0–10, défaut pour un 8B) ou `listwise` (RERANK_MODE)."""
    import os
    mode = mode or os.environ.get("RERANK_MODE", "pointwise")
    fused = list(fused)[:top]
    norm = dict(normalize_scores(fused))
    ids = [i for i, _ in fused if i in cards]
    llm_scores: dict[str, tuple[float, str]] = {}
    if mode == "listwise":
        llm_scores = _rank_listwise(description, [(i, _short(cards[i], 400)) for i in ids], model)
    else:
        for start in range(0, len(ids), batch_size):
            batch = [(i, _short(cards[i])) for i in ids[start : start + batch_size]]
            llm_scores.update(_score_batch(description, batch, model))
    out = []
    for i, _ in fused:
        ls = llm_scores.get(i)
        llm_part = ls[0] / 10.0 if ls else norm.get(i, 0.0)   # sans note LLM : on retombe sur la fusion
        final = alpha * norm.get(i, 0.0) + (1 - alpha) * llm_part
        out.append(Reranked(id=i, fused=norm.get(i, 0.0), llm_score=ls[0] if ls else None, reason=ls[1] if ls else "", final=final))
    out.sort(key=lambda r: -r.final)
    return out


def _short(card: str, max_chars: int = 600) -> str:
    """Fiche courte : on coupe le résumé long pour tenir 5 fiches dans le contexte."""
    return card if len(card) <= max_chars else card[:max_chars].rsplit(" ", 1)[0] + "…"
