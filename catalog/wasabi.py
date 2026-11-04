"""Loading and filtering of the WASABI Song Corpus 2.0.

Source: Zenodo, DOI 10.5281/zenodo.5603369, file `wasabi-2-0.tar` (2.7 GB),
licence CC BY-NC 4.0 (non-commercial use, cite Buffa et al., ESWC 2021).
The JSON dump (`json/json.zip`) contains `song.json` (13 GB, indented JSON
array, 2.1 M songs), `album.json`, `artist-without-members.json`,
`artist-members.json`, plus LastFM tags and emotions.

This module **streams** these files (ijson, C backend) and keeps only the
useful metadata in Parquet files under `data/catalog/wasabi/`. The lyrics
(`lyrics`) and the `summary` field (lines extracted from the lyrics) are never
kept, in line with the project rules (no stored lyrics).

Finding (checked on the dump): Deezer ids are present
(`id_song_deezer`, `urlDeezer`, `preview`) on the vast majority of songs,
as well as `isrc`, `id_song_musicbrainz`, `rank` (Deezer popularity) and
`language_detect`. Deezer search will only be used as a fallback.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import get_settings

logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/5603369/files/wasabi-2-0.tar?download=1"
ZENODO_MD5 = "90822a9d41b6bc4c4793372418d847b4"

_ABSTRACT_RE = re.compile(r"<dbo:abstract[^>]*xml:lang='en'[^>]*>(.*?)</dbo:abstract>", re.DOTALL)
_YEAR_RE = re.compile(r"(1[89]\d\d|20\d\d)")


def raw_dir() -> Path:
    return get_settings().data_dir / "raw" / "wasabi"


def out_dir() -> Path:
    d = get_settings().data_dir / "catalog" / "wasabi"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- utilities

def _oid(v: Any) -> str | None:
    if isinstance(v, dict):
        return v.get("$oid")
    return str(v) if v else None


def _year(*candidates: Any) -> int | None:
    for c in candidates:
        if not c:
            continue
        m = _YEAR_RE.search(str(c))
        if m:
            return int(m.group(1))
    return None


def _int(v: Any) -> int | None:
    try:
        return int(str(v)) if v not in (None, "") else None
    except ValueError:
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _text(v: Any, max_chars: int = 1000) -> str | None:
    """Text field possibly given as a list of sentences (e.g. `summary`)."""
    if v in (None, "", []):
        return None
    if isinstance(v, list):
        v = " ".join(str(x).strip() for x in v if x)
    v = re.sub(r"\s+", " ", str(v)).strip()
    return v[:max_chars] if v else None


def _abstract(rdf: str | None, max_chars: int = 600) -> str | None:
    """DBpedia abstract (English) embedded in the `rdf` field, cleaned and truncated."""
    if not rdf:
        return None
    m = _ABSTRACT_RE.search(rdf)
    if not m:
        return None
    text = html.unescape(m.group(1)).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars] if text else None


def _stream(path: Path, tolerate_truncated: bool = False) -> Iterator[dict]:
    """Iterates over the objects of a JSON array without loading the file."""
    with path.open("rb") as f:
        try:
            yield from ijson.items(f, "item", use_float=True)
        except ijson.common.IncompleteJSONError:
            if not tolerate_truncated:
                raise
            logger.warning("truncated JSON (sample?): %s", path.name)


# --------------------------------------------------------------------------- document projection

SONG_SCHEMA = pa.schema([
    ("wasabi_id", pa.string()),
    ("title", pa.string()),
    ("artist", pa.string()),
    ("album_title", pa.string()),
    ("album_id", pa.string()),
    ("year", pa.int32()),
    ("deezer_year", pa.int32()),
    ("album_genre", pa.string()),
    ("language", pa.string()),
    ("deezer_id", pa.int64()),
    ("deezer_artist_id", pa.int64()),
    ("deezer_album_id", pa.int64()),
    ("isrc", pa.string()),
    ("musicbrainz_id", pa.string()),
    ("rank", pa.int64()),
    ("bpm", pa.float32()),
    ("length_s", pa.int32()),
    ("preview_url", pa.string()),
    ("available_fr", pa.bool_()),
    ("n_countries", pa.int16()),
    ("url_wikipedia", pa.string()),
    ("valence", pa.float32()),
    ("arousal", pa.float32()),
    ("explicit", pa.bool_()),
    ("has_social_tags", pa.bool_()),
])


def project_song(d: dict) -> dict:
    countries = d.get("availableCountries") or []
    lang = (d.get("language") or d.get("language_detect") or "") or None
    return {
        "wasabi_id": _oid(d.get("_id")),
        "title": d.get("title") or None,
        "artist": d.get("name") or None,
        "album_title": d.get("albumTitle") or None,
        "album_id": _oid(d.get("id_album")),
        # `publicationDateAlbum` = original release year (LyricsWikia); `publicationDate` =
        # date of the Deezer edition (often a reissue/remaster), kept separately.
        "year": _year(d.get("publicationDateAlbum"), d.get("publicationDate")),
        "deezer_year": _year(d.get("publicationDate")),
        "album_genre": d.get("album_genre") or None,
        "language": lang,
        "deezer_id": _int(d.get("id_song_deezer")),
        "deezer_artist_id": _int(d.get("id_artist_deezer")),
        "deezer_album_id": _int(d.get("id_album_deezer")),
        "isrc": d.get("isrc") or None,
        "musicbrainz_id": d.get("id_song_musicbrainz") or None,
        "rank": _int(d.get("rank")),
        "bpm": _float(d.get("bpm")),
        "length_s": _int(d.get("length")),
        "preview_url": d.get("preview") or None,
        "available_fr": "FR" in countries,
        "n_countries": len(countries),
        "url_wikipedia": d.get("urlWikipedia") or None,
        "valence": _float(d.get("valence_predicted") if d.get("valence") in (None, "") else d.get("valence")),
        "arousal": _float(d.get("arousal_predicted") if d.get("arousal") in (None, "") else d.get("arousal")),
        "explicit": bool(d.get("explicitLyrics")) if d.get("explicitLyrics") not in (None, "") else None,
        "has_social_tags": bool(d.get("has_social_tags")) if d.get("has_social_tags") is not None else None,
    }


ALBUM_SCHEMA = pa.schema([
    ("album_id", pa.string()),
    ("title", pa.string()),
    ("artist", pa.string()),
    ("artist_id", pa.string()),
    ("genre", pa.string()),
    ("year", pa.int32()),
    ("url_wikipedia", pa.string()),
    ("abstract", pa.string()),
])


def project_album(d: dict) -> dict:
    return {
        "album_id": _oid(d.get("_id")),
        "title": d.get("title") or None,
        "artist": d.get("name") or None,
        "artist_id": _oid(d.get("id_artist")),
        "genre": d.get("genre") or None,
        "year": _year(d.get("publicationDate")),
        "url_wikipedia": d.get("urlWikipedia") or None,
        "abstract": _abstract(d.get("rdf")),
    }


ARTIST_SCHEMA = pa.schema([
    ("artist_id", pa.string()),
    ("name", pa.string()),
    ("genres", pa.list_(pa.string())),
    ("location", pa.string()),
    ("url_wikipedia", pa.string()),
    ("abstract", pa.string()),
    ("has_members", pa.bool_()),
])


def project_artist(d: dict, has_members: bool) -> dict:
    loc = d.get("locationInfo") or []
    return {
        "artist_id": _oid(d.get("_id")),
        "name": d.get("name") or None,
        "genres": [g for g in (d.get("genres") or []) if isinstance(g, str)],
        "location": ", ".join(str(x) for x in loc) if loc else None,
        "url_wikipedia": d.get("urlWikipedia") or None,
        "abstract": _abstract(d.get("rdf")),
        "has_members": has_members,
    }


# --------------------------------------------------------------------------- extraction

def _write_parquet(rows: Iterator[dict], schema: pa.Schema, out: Path, label: str,
                   batch_size: int = 50_000) -> int:
    tmp = out.with_suffix(".parquet.tmp")
    n = 0
    batch: list[dict] = []
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                w.write_table(pa.Table.from_pylist(batch, schema=schema))
                n += len(batch)
                batch = []
                logger.info("%s: %d documents", label, n)
        if batch:
            w.write_table(pa.Table.from_pylist(batch, schema=schema))
            n += len(batch)
    tmp.replace(out)
    logger.info("%s: %d documents written to %s", label, n, out)
    return n


def extract(src: Path | None = None, dst: Path | None = None, tolerate_truncated: bool = False,
            force: bool = False) -> dict[str, int]:
    """Projects song/album/artist from the JSON dump to three Parquet files. Resumable: an
    existing file is not recomputed (unless `force`)."""
    src = src or (raw_dir() / "json")
    dst = dst or out_dir()
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    jobs = [
        ("songs", [("song.json", lambda d: project_song(d))], SONG_SCHEMA),
        ("albums", [("album.json", lambda d: project_album(d))], ALBUM_SCHEMA),
        ("artists", [("artist-without-members.json", lambda d: project_artist(d, False)),
                     ("artist-members.json", lambda d: project_artist(d, True))], ARTIST_SCHEMA),
    ]
    for name, sources, schema in jobs:
        out = dst / f"{name}.parquet"
        if out.exists() and not force:
            counts[name] = pq.read_metadata(out).num_rows
            logger.info("%s already extracted (%d rows), skipped", name, counts[name])
            continue
        missing = [f for f, _ in sources if not (src / f).exists()]
        if missing:
            raise FileNotFoundError(f"missing files in {src}: {missing}")

        def rows() -> Iterator[dict]:
            for fname, proj in sources:
                for d in _stream(src / fname, tolerate_truncated):
                    yield proj(d)

        counts[name] = _write_parquet(rows(), schema, out, name)
    return counts


# --------------------------------------------------------------------------- reading

def load_songs(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "songs.parquet", columns=columns)


def load_albums() -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "albums.parquet")


def load_artists() -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "artists.parquet")
