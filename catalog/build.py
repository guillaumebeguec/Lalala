"""Assembly of the mainstream catalogue (PLAN.md phase 1.3).

Sources, in priority order:
  a. **tot**      : all ToT_Music answer tracks (WASABI → Deezer via ISRC);
  b. **hits_intl** / **hits_fr** : Deezer editorial playlists per year 1970–2025
                    ("Top Hits YYYY" by Deezer Best Of; "YYYY. Les 50 meilleurs Tubes
                    de l'année en France" and "Best of YYYY Français" by Top Hits France);
  c. **ads**      : `data/manual/ads_series_fr.csv` (titre, artiste, usage, annee), resolved
                    by Deezer search with a similarity check;
  d. **distract** : other titles by the same artists (WASABI, best Deezer `rank`),
                    then same genres, up to the target.

Each track is enriched from WASABI (original year, genre, language, MusicBrainz)
when its ISRC or Deezer id is found there. Deduplication on normalised (title, artist):
the best-ranked version is kept, ToT answers taking priority.

Output: `data/catalog/catalog.parquet` (+ `catalog_dev.parquet`, dev subset of
2,000 tracks including 300 ToT answers). No audio data here.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from catalog import wasabi
from catalog.deezer import DeezerClient, track_summary
from catalog.normalize import dedup_key, is_version_of, norm_artist, similarity
from core.config import get_settings
from testsets import tot_music

logger = logging.getLogger(__name__)

YEARS = range(1970, 2026)
TARGET_SIZE = 25_000
DEV_SIZE = 2_000
DEV_TOT = 300
SEED = 20261028

PRIORITY = {"tot": 0, "ads": 1, "hits_fr": 2, "hits_intl": 3, "distract": 4}


def catalog_dir() -> Path:
    d = get_settings().data_dir / "catalog"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Track:
    title: str
    artist: str
    deezer_id: int | None = None
    isrc: str | None = None
    wasabi_id: str | None = None
    album_title: str | None = None
    year: int | None = None          # original release year (WASABI) or playlist year
    deezer_year: int | None = None
    genre: str | None = None
    language: str | None = None
    musicbrainz_id: str | None = None
    rank: int | None = None
    preview_url: str | None = None
    deezer_artist_id: int | None = None
    usage: str | None = None         # ad / theme tune (manual list)
    sources: list[str] = field(default_factory=list)
    chart_years: list[int] = field(default_factory=list)
    country_hint: str | None = None  # "FR" if from a France playlist

    @property
    def track_id(self) -> str:
        if self.deezer_id:
            return f"dz{self.deezer_id}"
        return f"wsb{self.wasabi_id}"

    @property
    def priority(self) -> tuple:
        return (min(PRIORITY[s] for s in self.sources), -(self.rank or 0))

    def merge(self, other: "Track") -> None:
        for s in other.sources:
            if s not in self.sources:
                self.sources.append(s)
        self.chart_years = sorted(set(self.chart_years) | set(other.chart_years))
        for attr in ("deezer_id", "isrc", "wasabi_id", "album_title", "year", "deezer_year", "genre",
                     "language", "musicbrainz_id", "preview_url", "deezer_artist_id", "usage", "country_hint"):
            if getattr(self, attr) is None and getattr(other, attr) is not None:
                setattr(self, attr, getattr(other, attr))
        self.rank = max(self.rank or 0, other.rank or 0) or None


# --------------------------------------------------------------------------- WASABI

_LANG_MAP = {"eng": "english", "fra": "french", "fre": "french", "spa": "spanish", "ger": "german", "deu": "german",
             "ita": "italian", "por": "portuguese"}


def _norm_lang(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return None
    v = str(v).lower()
    return _LANG_MAP.get(v, v)


class WasabiIndex:
    """Fast access to WASABI songs by ISRC, Deezer id and artist."""

    def __init__(self, songs: pd.DataFrame | None = None):
        s = songs if songs is not None else wasabi.load_songs()
        s = s[s.title.notna() & s.artist.notna()].copy()
        s["language"] = s.language.map(_norm_lang)
        self.songs = s
        self.by_isrc = s[s.isrc.notna()].drop_duplicates("isrc").set_index("isrc", drop=False)
        self.by_deezer = s[s.deezer_id.notna()].drop_duplicates("deezer_id").set_index("deezer_id", drop=False)
        self.by_id = s.set_index("wasabi_id", drop=False)
        self._by_artist: dict[str, pd.DataFrame] | None = None

    def lookup(self, isrc: str | None = None, deezer_id: int | None = None, wasabi_id: str | None = None):
        if wasabi_id and wasabi_id in self.by_id.index:
            return self.by_id.loc[wasabi_id]
        if isrc and isrc in self.by_isrc.index:
            return self.by_isrc.loc[isrc]
        if deezer_id and deezer_id in self.by_deezer.index:
            return self.by_deezer.loc[deezer_id]
        return None

    def enrich(self, t: Track) -> bool:
        row = self.lookup(t.isrc, t.deezer_id, t.wasabi_id)
        if row is None:
            return False
        t.wasabi_id = t.wasabi_id or row["wasabi_id"]
        t.isrc = t.isrc or (row.isrc if isinstance(row.isrc, str) else None)
        t.deezer_id = t.deezer_id or (int(row.deezer_id) if pd.notna(row.deezer_id) else None)
        t.album_title = t.album_title or (row.album_title if isinstance(row.album_title, str) else None)
        if pd.notna(row.year):
            t.year = int(row.year) if t.year is None or int(row.year) < t.year else t.year
        t.deezer_year = t.deezer_year or (int(row.deezer_year) if pd.notna(row.deezer_year) else None)
        t.genre = t.genre or (row.album_genre if isinstance(row.album_genre, str) else None)
        t.language = t.language or row.language
        t.musicbrainz_id = t.musicbrainz_id or (row.musicbrainz_id if isinstance(row.musicbrainz_id, str) else None)
        t.preview_url = t.preview_url or (row.preview_url if isinstance(row.preview_url, str) else None)
        t.rank = max(t.rank or 0, int(row["rank"]) if pd.notna(row["rank"]) else 0) or None
        return True

    def tracks_of_artist(self, artist: str, limit: int, exclude: set[str]) -> list[Track]:
        if self._by_artist is None:
            s = self.songs[self.songs.deezer_id.notna() & self.songs.available_fr]
            s = s.assign(_a=s.artist.map(norm_artist))
            self._by_artist = {k: g.sort_values("rank", ascending=False) for k, g in s.groupby("_a")}
        g = self._by_artist.get(norm_artist(artist))
        if g is None:
            return []
        out: list[Track] = []
        for _, r in g.iterrows():
            if is_version_of(r.title):
                continue
            k = dedup_key(r.title, r.artist)
            if k in exclude:
                continue
            exclude.add(k)
            out.append(Track(title=r.title, artist=r.artist, deezer_id=int(r.deezer_id),
                             isrc=r.isrc if isinstance(r.isrc, str) else None, wasabi_id=r.wasabi_id,
                             album_title=r.album_title if isinstance(r.album_title, str) else None,
                             year=int(r.year) if pd.notna(r.year) else None,
                             genre=r.album_genre if isinstance(r.album_genre, str) else None,
                             language=r.language, rank=int(r["rank"]), sources=["distract"]))
            if len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- sources

def _from_deezer(t: dict, source: str, chart_year: int | None = None, country: str | None = None) -> Track:
    s = track_summary(t)
    return Track(title=s["title"] or "", artist=s["artist"] or "", deezer_id=s["deezer_id"], isrc=s["isrc"],
                 album_title=s["album_title"], rank=s["rank"], preview_url=s["preview_url"],
                 deezer_artist_id=s["deezer_artist_id"], year=None, sources=[source],
                 chart_years=[chart_year] if chart_year else [], country_hint=country)


def source_tot(dz: DeezerClient, widx: WasabiIndex, resolve: bool = True) -> list[Track]:
    """ToT_Music answers: WASABI metadata, current Deezer id via ISRC."""
    out: list[Track] = []
    n_resolved = 0
    for wid, a in tot_music.answer_ids().items():
        t = Track(title=a["title"], artist=", ".join(a["artists"]), wasabi_id=wid, isrc=a["isrc"] or None, sources=["tot"])
        row = widx.lookup(wasabi_id=wid)
        if row is not None:
            t.title, t.artist = row.title, row.artist
        widx.enrich(t)
        if resolve and t.isrc:
            live = dz.track_by_isrc(t.isrc)
            if live:
                t.deezer_id = live["id"]
                t.preview_url = live.get("preview") or t.preview_url
                t.rank = max(t.rank or 0, live.get("rank") or 0) or None
                n_resolved += 1
        out.append(t)
    logger.info("tot: %d tracks, %d resolved on Deezer via ISRC", len(out), n_resolved)
    return out


def discover_playlists(dz: DeezerClient) -> dict[str, dict[int, int]]:
    """Editorial playlists per year. Result: {"intl": {year: id}, "fr": {year: id}}."""
    intl, fr = {}, {}
    for y in YEARS:
        for p in dz.search_playlists(f"Top Hits {y}", limit=10):
            if p["user"]["name"] == "Deezer Best Of" and p["title"].strip() == f"Top Hits {y}":
                intl[y] = p["id"]
                break
        if y not in intl:
            for p in dz.search_playlists(f"Best of Hits {y}", limit=10):
                if p["user"]["name"] == "Deezer Best Of" and str(y) in p["title"] and p["nb_tracks"] >= 30:
                    intl[y] = p["id"]
                    break
        for q, pred in ((f"{y} meilleurs tubes de l'année en France", lambda t: t.startswith(f"{y}.") and "France" in t),
                        (f"Best of {y} Français", lambda t: f"Best of {y} Français" in t)):
            for p in dz.search_playlists(q, limit=10):
                if p["user"]["name"].startswith("Top Hits France") and pred(p["title"]):
                    fr[y] = p["id"]
                    break
            if y in fr:
                break
    logger.info("playlists: intl %d/%d years, fr %d/%d years; missing intl=%s fr=%s",
                len(intl), len(YEARS), len(fr), len(YEARS),
                [y for y in YEARS if y not in intl], [y for y in YEARS if y not in fr])
    return {"intl": intl, "fr": fr}


def source_hits(dz: DeezerClient, playlists: dict[str, dict[int, int]]) -> list[Track]:
    out: list[Track] = []
    for kind, by_year in playlists.items():
        src = f"hits_{kind}"
        for y, pid in sorted(by_year.items()):
            tracks = dz.playlist_tracks(pid)
            for t in tracks:
                if not t.get("readable", True):
                    continue
                out.append(_from_deezer(t, src, chart_year=y, country="FR" if kind == "fr" else None))
            logger.info("%s %d: %d tracks (playlist %d)", src, y, len(tracks), pid)
    return out


def source_ads(dz: DeezerClient, path: Path | None = None, min_sim: float = 0.8) -> list[Track]:
    """Manual list of ads / theme tunes, resolved by Deezer search with a similarity check."""
    path = path or (get_settings().manual_dir / "ads_series_fr.csv")
    if not path.exists():
        logger.warning("ads: %s missing", path)
        return []
    out: list[Track] = []
    rejected: list[str] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title, artist = (row.get("titre") or "").strip(), (row.get("artiste") or "").strip()
            if not title or not artist:
                continue
            best, best_score = None, 0.0
            for cand in dz.search_track(artist, title, limit=10):
                sc = min(similarity(cand["title"], title) + 0.0, 1.0) * 0.6 + similarity(cand["artist"]["name"], artist) * 0.4
                if is_version_of(cand["title"]) and not is_version_of(title):
                    sc -= 0.15
                if sc > best_score:
                    best, best_score = cand, sc
            if best is None or best_score < min_sim:
                rejected.append(f"{title} — {artist} (score {best_score:.2f})")
                continue
            t = _from_deezer(best, "ads", country="FR")
            t.usage = (row.get("usage") or "").strip() or None
            y = (row.get("annee") or "").strip()
            t.year = int(y) if y.isdigit() else None
            out.append(t)
    logger.info("ads: %d resolved, %d rejected", len(out), len(rejected))
    for r in rejected:
        logger.warning("ads unmatched: %s", r)
    return out


# --------------------------------------------------------------------------- assembly

def dedup(tracks: Iterable[Track]) -> dict[str, Track]:
    by_key: dict[str, Track] = {}
    for t in tracks:
        if not t.title or not t.artist:
            continue
        k = dedup_key(t.title, t.artist)
        if k in by_key:
            a, b = by_key[k], t
            keep, other = (a, b) if a.priority <= b.priority else (b, a)
            keep.merge(other)
            by_key[k] = keep
        else:
            by_key[k] = t
    return by_key


def add_distractors(by_key: dict[str, Track], widx: WasabiIndex, target: int, per_artist: int = 6,
                    seed: int = SEED) -> None:
    rng = random.Random(seed)
    seeds = [t for t in by_key.values() if "distract" not in t.sources]
    artists = sorted({t.artist for t in seeds})
    rng.shuffle(artists)
    exclude = set(by_key)
    n0 = len(by_key)
    for a in artists:
        if len(by_key) >= target:
            break
        for t in widx.tracks_of_artist(a, per_artist, exclude):
            by_key[dedup_key(t.title, t.artist)] = t
    logger.info("same-artist distractors: +%d", len(by_key) - n0)
    if len(by_key) < target:
        n1 = len(by_key)
        genres = Counter(t.genre for t in seeds if t.genre)
        pool = widx.songs[widx.songs.deezer_id.notna() & widx.songs.available_fr & widx.songs.album_genre.isin(genres)]
        pool = pool[pool["rank"] >= pool["rank"].quantile(0.9)]
        for _, r in pool.sample(frac=1.0, random_state=seed).iterrows():
            if len(by_key) >= target:
                break
            k = dedup_key(r.title, r.artist)
            if k in exclude or is_version_of(r.title):
                continue
            exclude.add(k)
            by_key[k] = Track(title=r.title, artist=r.artist, deezer_id=int(r.deezer_id),
                              isrc=r.isrc if isinstance(r.isrc, str) else None, wasabi_id=r.wasabi_id,
                              year=int(r.year) if pd.notna(r.year) else None,
                              genre=r.album_genre if isinstance(r.album_genre, str) else None,
                              language=r.language, rank=int(r["rank"]), sources=["distract"])
        logger.info("same-genre distractors: +%d", len(by_key) - n1)


def to_frame(tracks: Iterable[Track]) -> pd.DataFrame:
    rows = []
    for t in tracks:
        rows.append({
            "track_id": t.track_id, "deezer_id": t.deezer_id, "isrc": t.isrc, "wasabi_id": t.wasabi_id,
            "title": t.title, "artist": t.artist, "album_title": t.album_title,
            "year": t.year if t.year else (min(t.chart_years) if t.chart_years else None),
            "year_source": "meta" if t.year else ("chart" if t.chart_years else None),
            "deezer_year": t.deezer_year, "genre": t.genre, "language": t.language,
            "musicbrainz_id": t.musicbrainz_id, "rank": t.rank, "preview_url": t.preview_url,
            "deezer_artist_id": t.deezer_artist_id, "usage": t.usage, "sources": t.sources,
            "chart_years": t.chart_years, "country_hint": t.country_hint,
            "is_tot_answer": "tot" in t.sources,
        })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("track_id").reset_index(drop=True)
    return df


def make_dev_subset(df: pd.DataFrame, size: int = DEV_SIZE, n_tot: int = DEV_TOT, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    tot_idx = df.index[df.is_tot_answer].to_numpy()
    chosen = set(rng.choice(tot_idx, size=min(n_tot, len(tot_idx)), replace=False).tolist())
    rest = df.index[~df.index.isin(chosen) & ~df.is_tot_answer].to_numpy()
    # the rest is taken outside the ToT answers, all sources combined (hits, ads, distractors)
    chosen |= set(rng.choice(rest, size=min(size - len(chosen), len(rest)), replace=False).tolist())
    return df.index.isin(chosen)


def build(target: int = TARGET_SIZE, resolve_tot: bool = True, with_distractors: bool = True) -> pd.DataFrame:
    dz = DeezerClient()
    widx = WasabiIndex()
    tracks: list[Track] = []
    tracks += source_tot(dz, widx, resolve=resolve_tot)
    playlists = discover_playlists(dz)
    (catalog_dir() / "playlists.json").write_text(json.dumps(playlists, indent=1))
    hits = source_hits(dz, playlists)
    ads = source_ads(dz)
    for t in hits + ads:
        widx.enrich(t)
    tracks += hits + ads
    by_key = dedup(tracks)
    logger.info("after deduplicating the sources: %d tracks", len(by_key))
    if with_distractors:
        add_distractors(by_key, widx, target)
    df = to_frame(by_key.values())
    df["in_dev"] = make_dev_subset(df)
    df.to_parquet(catalog_dir() / "catalog.parquet", index=False)
    df[df.in_dev].to_parquet(catalog_dir() / "catalog_dev.parquet", index=False)
    logger.info("catalogue: %d tracks (%d dev); Deezer requests: %d, cache: %d",
                len(df), int(df.in_dev.sum()), dz.n_requests, dz.n_cached)
    return df


def load(subset: str = "full") -> pd.DataFrame:
    name = "catalog_dev.parquet" if subset == "dev" else "catalog.parquet"
    return pd.read_parquet(catalog_dir() / name)
