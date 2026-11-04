"""T4 — Covers → original (PLAN.md phase 2.4).

For N catalog hits (best `rank`, excluding ToT answers so as not to bias T1), we
search Deezer for "title piano", "title acoustic cover", "title live",
"title karaoke". We keep results whose normalised title matches the original
(similarity ≥ 0.85) and whose artist **differs** (or whose title mentions
"live"), with a preview. Metadata only; the audio is read into memory at
evaluation time (see catalog/previews.py).

Output: data/testsets/t4_covers.jsonl — fields: id, cover_deezer_id,
cover_title, cover_artist, cover_preview_url, kind, reponse_id (original).
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from catalog.deezer import DeezerClient, track_summary
from catalog.normalize import is_version_of, norm_artist, norm_title, similarity
from testsets.common import SEED, testsets_dir, write_jsonl

logger = logging.getLogger(__name__)

QUERIES = {"piano": "{title} piano", "acoustic": "{title} acoustic cover", "live": "{title} live", "karaoke": "{title} karaoke"}


def find_covers(dz: DeezerClient, title: str, artist: str, per_kind: int = 2) -> list[dict]:
    out, seen = [], set()
    nt, na = norm_title(title), norm_artist(artist)
    for kind, tpl in QUERIES.items():
        n_kind = 0
        for c in dz.search_tracks(tpl.format(title=title), limit=15):
            if not c.get("preview") or not c.get("readable", True) or c["id"] in seen:
                continue
            ct, ca = c.get("title") or "", (c.get("artist") or {}).get("name") or ""
            if similarity(norm_title(ct), nt) < 0.85:
                continue
            same_artist = norm_artist(ca) == na or similarity(norm_artist(ca), na) > 0.9
            if same_artist and kind != "live":
                continue
            if same_artist and kind == "live" and not is_version_of(ct):
                continue
            if kind == "karaoke" and "karaoke" not in (ct + " " + ca).lower():
                continue
            seen.add(c["id"])
            s = track_summary(c)
            out.append({"cover_deezer_id": s["deezer_id"], "cover_title": s["title"], "cover_artist": s["artist"],
                        "cover_preview_url": s["preview_url"], "kind": kind, "same_artist": same_artist})
            n_kind += 1
            if n_kind >= per_kind:
                break
    return out


def build_t4(catalog: pd.DataFrame, previews: pd.DataFrame | None, n_hits: int = 300, target: int = 200,
             seed: int = SEED) -> list[dict]:
    ok = set(previews[previews.status == "ok"].track_id) if previews is not None else None
    pool = catalog[~catalog.is_tot_answer].copy()
    if ok is not None:
        pool = pool[pool.track_id.isin(ok)]
    pool = pool[~pool.title.map(is_version_of)].sort_values("rank", ascending=False).head(n_hits)
    dz = DeezerClient()
    rows = []
    for i, r in enumerate(pool.itertuples(index=False), 1):
        for c in find_covers(dz, r.title, r.artist):
            rows.append({"id": f"t4_{len(rows) + 1:04d}", **c, "reponse_id": r.track_id,
                         "original_title": r.title, "original_artist": r.artist})
        if i % 50 == 0:
            logger.info("T4: %d hits explored, %d covers", i, len(rows))
    rng = random.Random(seed)
    rng.shuffle(rows)
    # at most 2 covers per original for diversity, then the target size
    per: dict[str, int] = {}
    kept = []
    for r in rows:
        if per.get(r["reponse_id"], 0) >= 2:
            continue
        per[r["reponse_id"]] = per.get(r["reponse_id"], 0) + 1
        kept.append(r)
    kept = kept[:target]
    logger.info("T4: %d cover → original pairs (%d originals); Deezer requests %d", len(kept), len(per), dz.n_requests)
    return kept


def write_t4(catalog: pd.DataFrame, previews: pd.DataFrame | None, **kw):
    rows = build_t4(catalog, previews, **kw)
    out = testsets_dir() / "t4_covers.jsonl"
    write_jsonl(out, rows)
    return out
