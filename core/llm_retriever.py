"""Llama comme moteur de recherche à part entière (recommandation n° 1).

Pour un souvenir décrit en mots, le meilleur signal est souvent la connaissance du modèle :
on lui demande « quelles chansons correspondent à cette description ? » (titre + artiste +
année, avec une confiance), puis on **résout** chaque proposition dans le catalogue (titre /
artiste normalisés, similarité ≥ 0,85). Les titres non résolus sont renvoyés à part
(candidats « hors catalogue », à vérifier par le web), jamais injectés comme certitudes.

Un Llama 8B connaît mal les morceaux ; un modèle plus gros via API (LLM_BASE_URL / LLM_MODEL)
change l'échelle : `LLM_RETRIEVER_MODEL` permet d'utiliser un modèle différent pour cette
seule étape. Cache par hash de la description.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from core import llm
from core.cache import DiskCache
from core.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = {
    "title": "llm_candidates",
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array", "maxItems": 10,
            "items": {"type": "object", "required": ["title", "artist", "confidence"],
                      "properties": {"title": {"type": "string"}, "artist": {"type": "string"},
                                     "year": {"type": ["integer", "null"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                     "why": {"type": "string"}}},
        }
    },
}

_SYSTEM = (
    "You are a music expert with encyclopedic knowledge of popular songs (1960–2025, international and French). "
    "Someone describes a song they half-remember (lyrics fragments, story, era, genre, where they heard it: ad, film, series). "
    "List up to 8 real songs that match, best first, each with title, artist, year and a confidence between 0 and 1. "
    "Only real, existing songs. If you are not sure, lower the confidence rather than inventing. "
    "JSON only: {\"candidates\": [{\"title\": ..., \"artist\": ..., \"year\": ..., \"confidence\": ..., \"why\": \"<10 words\"}]}"
)


@dataclass
class LLMCandidate:
    title: str
    artist: str
    year: int | None
    confidence: float
    why: str


def propose(description: str, model: str | None = None, max_items: int = 8) -> list[LLMCandidate]:
    model = model or os.environ.get("LLM_RETRIEVER_MODEL") or get_settings().llm_model
    cache = DiskCache("llm_retriever")
    key = cache.key(description, model=model, v=1)
    hit = cache.get(key)
    if hit is None:
        try:
            r = llm.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": description.strip()}],
                         json_schema=SCHEMA, temperature=0.2, max_tokens=700, model=model)
            hit = r.json or {"candidates": []}
        except llm.LLMError as e:
            logger.warning("llm_retriever : %s", e)
            hit = {"candidates": []}
        cache.put(key, hit)
    out: list[LLMCandidate] = []
    seen: set[tuple[str, str]] = set()
    for c in (hit.get("candidates") or []):
        if len(out) >= max_items:
            break
        if not isinstance(c, dict) or not c.get("title") or not c.get("artist"):
            continue
        k = (str(c["title"]).strip().lower(), str(c["artist"]).strip().lower())
        if k in seen:   # le 8B répète souvent ses propositions
            continue
        seen.add(k)
        try:
            conf = float(c.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        y = c.get("year")
        out.append(LLMCandidate(title=str(c["title"]).strip(), artist=str(c["artist"]).strip(),
                                year=int(y) if isinstance(y, (int, float)) and 1900 < int(y) < 2100 else None,
                                confidence=max(0.0, min(1.0, conf)), why=str(c.get("why") or "")[:120]))
    return out
