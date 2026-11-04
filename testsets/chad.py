"""T6a — CHAD hums → catalog original (PLAN_COMPLEMENTS.md C3).

Source: CHAD (Amatov et al., ISMIR 2023), repository `amanteur/CHAD` (MIT code); hums subset
`amanteur/CHAD_hummings` on Hugging Face (CC BY-NC 4.0): 5,417 wav files, 693 fragments,
311 tracks (groups). Originals and covers are referenced only by YouTube id + interval:
**no YouTube audio is downloaded**; only the public video title is read (oEmbed endpoint,
metadata) to find the track in our catalog (normalised title/artist, similarity ≥ 0.85).
T6b (CHAD covers) is not built: it would require YouTube audio.

Documented limitation: the hummed fragment corresponds to an interval of the full track
(`interval`), whereas the Deezer preview covers 30 s at an unknown position. The fragment
interval is carried over so results can be analysed by position.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
from pathlib import Path

import httpx
import pandas as pd

from catalog.normalize import dedup_key, norm_artist, norm_title, similarity
from core.cache import DiskCache
from core.config import get_settings
from testsets.common import testsets_dir, write_jsonl

logger = logging.getLogger(__name__)

_SEP_RE = re.compile(r"\s+[-–—|:]\s+")
_NOISE_RE = re.compile(r"\s*[\(\[][^\)\]]*(official|video|audio|lyrics?|hd|hq|remaster|live|clip|version|visualizer|4k)[^\)\]]*[\)\]]", re.IGNORECASE)
_TAIL_RE = re.compile(r"\s*(official (music )?video|official audio|lyrics? video|lyrics|hd|hq)\s*$", re.IGNORECASE)



def _rel(p) -> str:
    """Path stored in the test sets: relative to the project root when inside it (portable across machines)."""
    from pathlib import Path
    from core.config import PROJECT_ROOT
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

def chad_dir() -> Path:
    return get_settings().data_dir / "raw" / "chad"


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(chad_dir() / "repo" / "metadata" / "dataset.csv", dtype=str)
    df["interval"] = df.interval.map(lambda s: tuple(ast.literal_eval(s)) if isinstance(s, str) else None)
    return df


def humming_files() -> dict[str, Path]:
    """{id: wav path} of the hums present locally."""
    out = {}
    for p in (chad_dir() / "hummings").rglob("*.wav"):
        out[p.stem] = p
    return out


def youtube_title(video_id: str, cache: DiskCache | None = None, client: httpx.Client | None = None) -> str | None:
    """Public title of a video via oEmbed (metadata only, no audio)."""
    cache = cache or DiskCache("youtube_oembed")
    key = cache.key(video_id)
    hit = cache.get(key)
    if hit is not None:
        return hit["v"]
    client = client or httpx.Client(timeout=15)
    title = None
    try:
        r = client.get("https://www.youtube.com/oembed", params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"})
        if r.status_code == 200:
            title = r.json().get("title")
    except httpx.HTTPError as e:
        logger.warning("oembed %s: %s", video_id, e)
    cache.put(key, {"v": title})
    return title


def parse_artist_title(yt_title: str) -> tuple[str, str] | None:
    """\"Artist - Title (Official Video)\" → (artist, title). None if there is no separator."""
    t = _NOISE_RE.sub("", yt_title)
    t = _TAIL_RE.sub("", t).strip(" -–|")
    parts = _SEP_RE.split(t, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    b = re.sub(r"\s*\(.*?\)\s*$", "", b).strip() or b
    return (a, b) if a and b else None


def match_catalog(catalog: pd.DataFrame, artist: str, title: str, min_sim: float = 0.85) -> tuple[str | None, float]:
    if "_key" not in catalog.columns:
        catalog["_key"] = [dedup_key(t, a) for t, a in zip(catalog.title, catalog.artist)]
    for a, b in ((artist, title), (title, artist)):  # the "Title - Artist" order also exists
        hit = catalog[catalog["_key"] == dedup_key(b, a)]
        if len(hit):
            return hit.iloc[0].track_id, 1.0
    best, best_s = None, 0.0
    for a, b in ((artist, title), (title, artist)):
        na, nt = norm_artist(a), norm_title(b)
        cand = catalog[catalog.artist.map(norm_artist) == na]
        for r in cand.itertuples(index=False):
            s = similarity(norm_title(r.title), nt)
            if s > best_s:
                best, best_s = r.track_id, s
    return (best, best_s) if best_s >= min_sim else (None, best_s)


def build_t6a(catalog: pd.DataFrame, previews: pd.DataFrame | None, max_per_group: int = 3) -> tuple[list[dict], dict]:
    meta = load_metadata()
    files = humming_files()
    hums = meta[(meta.audio_type == "humming") & meta.id.isin(files)]
    groups = sorted(hums.group_id.unique())
    originals = meta[(meta.audio_type == "original") & meta.group_id.isin(groups)].drop_duplicates("group_id").set_index("group_id")
    ok = set(previews[previews.status == "ok"].track_id) if previews is not None else None
    cat = catalog.copy()
    cache = DiskCache("youtube_oembed")
    client = httpx.Client(timeout=15, headers={"User-Agent": "Lalala/0.1"})
    stats = {"groups": len(groups), "with_title": 0, "parsed": 0, "in_catalog": 0, "with_preview": 0, "cases": 0}
    group_track: dict[str, str] = {}
    group_interval: dict[str, tuple] = {}
    for g in groups:
        if g not in originals.index:
            continue
        o = originals.loc[g]
        title = youtube_title(o.youtube_id, cache, client)
        if not title:
            continue
        stats["with_title"] += 1
        parsed = parse_artist_title(title)
        if not parsed:
            continue
        stats["parsed"] += 1
        tid, _ = match_catalog(cat, *parsed)
        if not tid:
            continue
        stats["in_catalog"] += 1
        if ok is not None and tid not in ok:
            continue
        stats["with_preview"] += 1
        group_track[g] = tid
        group_interval[g] = o.interval
    rows = []
    for g, tid in group_track.items():
        for r in hums[hums.group_id == g].head(max_per_group).itertuples(index=False):
            rows.append({"id": f"t6a_{r.id}", "file": _rel(files[r.id]), "reponse_id": tid, "group_id": g, "fragment_id": r.fragment_id,
                         "duration_s": float(r.duration) if r.duration else None, "original_interval": list(group_interval.get(g) or []),
                         "kind": "humming"})
    stats["cases"] = len(rows)
    logger.info("T6a: %s", stats)
    return rows, stats


def write_t6a(catalog: pd.DataFrame, previews: pd.DataFrame | None) -> tuple[Path, dict]:
    rows, stats = build_t6a(catalog, previews)
    out = testsets_dir() / "t6a_chad_hummings.jsonl"
    write_jsonl(out, rows)
    (testsets_dir() / "t6a_stats.json").write_text(json.dumps(stats, indent=1))
    return out, stats
