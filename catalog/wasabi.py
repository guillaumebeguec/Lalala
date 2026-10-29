"""Chargement et filtrage du WASABI Song Corpus 2.0.

Source : Zenodo, DOI 10.5281/zenodo.5603369, fichier `wasabi-2-0.tar` (2,7 Go),
licence CC BY-NC 4.0 (usage non commercial, citer Buffa et al., ESWC 2021).
Le dump JSON (`json/json.zip`) contient `song.json` (13 Go, tableau JSON
indenté, 2,1 M chansons), `album.json`, `artist-without-members.json`,
`artist-members.json`, plus tags LastFM et émotions.

Ce module **streame** ces fichiers (ijson, backend C) et n'en garde que les
métadonnées utiles dans des Parquet sous `data/catalog/wasabi/`. Les paroles
(`lyrics`) et le champ `summary` (lignes extraites des paroles) ne sont jamais
conservés, conformément à CLAUDE.md.

Constat (vérifié sur le dump) : les identifiants Deezer sont présents
(`id_song_deezer`, `urlDeezer`, `preview`) sur la grande majorité des chansons,
ainsi que `isrc`, `id_song_musicbrainz`, `rank` (popularité Deezer) et
`language_detect`. La recherche Deezer ne servira qu'en repli.
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


# --------------------------------------------------------------------------- utilitaires

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
    """Champ texte éventuellement fourni comme liste de phrases (ex. `summary`)."""
    if v in (None, "", []):
        return None
    if isinstance(v, list):
        v = " ".join(str(x).strip() for x in v if x)
    v = re.sub(r"\s+", " ", str(v)).strip()
    return v[:max_chars] if v else None


def _abstract(rdf: str | None, max_chars: int = 600) -> str | None:
    """Résumé DBpedia (anglais) embarqué dans le champ `rdf`, nettoyé et tronqué."""
    if not rdf:
        return None
    m = _ABSTRACT_RE.search(rdf)
    if not m:
        return None
    text = html.unescape(m.group(1)).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars] if text else None


def _stream(path: Path, tolerate_truncated: bool = False) -> Iterator[dict]:
    """Itère sur les objets d'un tableau JSON, sans charger le fichier."""
    with path.open("rb") as f:
        try:
            yield from ijson.items(f, "item", use_float=True)
        except ijson.common.IncompleteJSONError:
            if not tolerate_truncated:
                raise
            logger.warning("JSON tronqué (échantillon ?) : %s", path.name)


# --------------------------------------------------------------------------- projection des documents

SONG_SCHEMA = pa.schema([
    ("wasabi_id", pa.string()),
    ("title", pa.string()),
    ("artist", pa.string()),
    ("album_title", pa.string()),
    ("album_id", pa.string()),
    ("year", pa.int32()),
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
        "year": _year(d.get("publicationDate"), d.get("publicationDateAlbum")),
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
                logger.info("%s : %d documents", label, n)
        if batch:
            w.write_table(pa.Table.from_pylist(batch, schema=schema))
            n += len(batch)
    tmp.replace(out)
    logger.info("%s : %d documents écrits dans %s", label, n, out)
    return n


def extract(src: Path | None = None, dst: Path | None = None, tolerate_truncated: bool = False,
            force: bool = False) -> dict[str, int]:
    """Projette song/album/artist du dump JSON vers trois Parquet. Reprise : un
    fichier déjà présent n'est pas recalculé (sauf `force`)."""
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
            logger.info("%s déjà extrait (%d lignes), passé", name, counts[name])
            continue
        missing = [f for f, _ in sources if not (src / f).exists()]
        if missing:
            raise FileNotFoundError(f"fichiers manquants dans {src} : {missing}")

        def rows() -> Iterator[dict]:
            for fname, proj in sources:
                for d in _stream(src / fname, tolerate_truncated):
                    yield proj(d)

        counts[name] = _write_parquet(rows(), schema, out, name)
    return counts


# --------------------------------------------------------------------------- lecture

def load_songs(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "songs.parquet", columns=columns)


def load_albums() -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "albums.parquet")


def load_artists() -> pd.DataFrame:
    return pd.read_parquet(out_dir() / "artists.parquet")
