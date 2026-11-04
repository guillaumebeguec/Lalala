"""T2 — Fuzzy memories in French (`data/manual/fr_memories.csv`: description, titre, artiste).

Validation: every answer must exist in the catalog (normalised title/artist match,
then similarity ≥ 0.85). Unmatched rows are listed for correction or for adding to
the catalog. Frozen output: data/testsets/t2_fr_memories.jsonl.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from catalog.normalize import dedup_key, norm_artist, norm_title, similarity
from core.config import get_settings
from testsets.common import testsets_dir, write_jsonl

logger = logging.getLogger(__name__)


def csv_path() -> Path:
    return get_settings().manual_dir / "fr_memories.csv"


def match_track(catalog: pd.DataFrame, title: str, artist: str, min_sim: float = 0.85) -> tuple[str | None, float]:
    key = dedup_key(title, artist)
    keys = catalog["_key"] if "_key" in catalog else catalog.apply(lambda r: dedup_key(r.title, r.artist), axis=1)
    hit = catalog[keys == key]
    if len(hit):
        return hit.iloc[0].track_id, 1.0
    nt, na = norm_title(title), norm_artist(artist)
    cand = catalog[catalog.artist.map(norm_artist) == na] if na else catalog.iloc[0:0]
    best, best_s = None, 0.0
    for r in cand.itertuples(index=False):
        s = similarity(norm_title(r.title), nt)
        if s > best_s:
            best, best_s = r.track_id, s
    return (best, best_s) if best_s >= min_sim else (None, best_s)


def build_t2(catalog: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    p = csv_path()
    rows, unmatched = [], []
    if not p.exists():
        logger.warning("T2: %s missing (to fill in: description, titre, artiste)", p)
        return rows, unmatched
    cat = catalog.assign(_key=catalog.apply(lambda r: dedup_key(r.title, r.artist), axis=1))
    with p.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            desc, title, artist = (r.get("description") or "").strip(), (r.get("titre") or "").strip(), (r.get("artiste") or "").strip()
            if not desc or not title:
                continue
            tid, s = match_track(cat, title, artist)
            if tid is None:
                unmatched.append({"ligne": i, "titre": title, "artiste": artist, "meilleure_similarite": round(s, 2)})
                continue
            rows.append({"id": f"t2_{i:03d}", "texte": desc, "reponse_id": tid, "titre": title, "artiste": artist})
    logger.info("T2: %d descriptions matched, %d unmatched", len(rows), len(unmatched))
    return rows, unmatched


def write_t2(catalog: pd.DataFrame) -> Path:
    rows, unmatched = build_t2(catalog)
    out = testsets_dir() / "t2_fr_memories.jsonl"
    write_jsonl(out, rows)
    for u in unmatched:
        logger.warning("T2 unmatched (line %d): %s — %s (sim %.2f)", u["ligne"], u["titre"], u["artiste"], u["meilleure_similarite"])
    return out
