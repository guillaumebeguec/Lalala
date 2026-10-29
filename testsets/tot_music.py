"""T1 — ToT_Music (Bhargav, Schuth, Hauff, SIGIR 2023), licence CC0-1.0.

Source : https://github.com/spotify-research/tot (fichier `data.jsonl.zip`, une
demande par ligne). 2 274 demandes r/TipOfMyTongue résolues, chacune reliée à
un morceau du corpus WASABI 2.0 (`wasabi_2.0_id`, `wasabi_isrc`) et à sa
résolution Spotify (titre, artistes).

Ce module ne fait que **récupérer et parser** : la constitution du jeu d'éval
(filtrage sur le catalogue, échantillon stratifié) est faite en phase 2.

Règle anti-fuite : `ToTQuery.text` et `ToTQuery.title` sont les seuls champs
qu'un agent ou un utilisateur simulé a le droit de voir. La réponse
(`answer_*`) ne doit jamais entrer dans un prompt.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/spotify-research/tot"
DATA_ZIP_URL = "https://github.com/spotify-research/tot/raw/main/data.jsonl.zip"

# Balises que les auteurs des posts ajoutent au titre : [TOMT][SONG][ROCK][80s]
_TAG_RE = re.compile(r"\[([^\]]+)\]")
_DECADE_RE = re.compile(r"(?<!\d)(?:19|20)?(\d0)'?s\b", re.IGNORECASE)


@dataclass
class ToTQuery:
    id: str
    title: str                      # titre nettoyé du post (visible par l'agent)
    text: str                       # corps nettoyé du post (visible par l'agent)
    raw_title: str                  # titre brut avec balises [TOMT][SONG]... (visible)
    query_types: list[str]
    split: str | None               # train / val / test / None (découpage de l'article)
    # ---- réponse : JAMAIS dans un prompt ----
    answer_wasabi_id: str
    answer_isrc: str
    answer_title: str
    answer_artists: list[str]
    answer_spotify_urls: list[str] = field(default_factory=list)
    # indices « utilisateur informé » dérivés des balises du titre (décennie)
    tag_decade: str | None = None
    tag_genre: str | None = None

    @property
    def visible_text(self) -> str:
        """Ce que voit l'agent : titre + texte du post, rien d'autre."""
        return f"{self.title}\n\n{self.text}".strip()


def raw_dir() -> Path:
    return get_settings().data_dir / "raw" / "tot"


def jsonl_path() -> Path:
    return raw_dir() / "data.jsonl"


def download(force: bool = False) -> Path:
    """Télécharge `data.jsonl.zip` depuis le dépôt et le décompresse dans data/raw/tot/."""
    out = jsonl_path()
    if out.exists() and not force:
        return out
    raw_dir().mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement de ToT_Music depuis %s", DATA_ZIP_URL)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        r = client.get(DATA_ZIP_URL)
        r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".jsonl")]
        if not names:
            raise RuntimeError("data.jsonl introuvable dans l'archive ToT_Music")
        out.write_bytes(zf.read(names[0]))
    logger.info("ToT_Music écrit dans %s", out)
    return out


def _parse_tags(raw_title: str) -> tuple[str | None, str | None]:
    decade = genre = None
    for tag in _TAG_RE.findall(raw_title or ""):
        t = tag.strip()
        up = t.upper()
        if up in {"TOMT", "SONG", "MUSIC", "SOLVED"}:
            continue
        m = _DECADE_RE.search(t)
        if m and decade is None:
            d = m.group(1)
            decade = ("20" if d in {"00", "10", "20"} else "19") + d + "s"
            continue
        if genre is None and len(t) <= 20 and not any(c.isdigit() for c in t):
            genre = t
    return decade, genre


def _parse_row(row: dict) -> ToTQuery:
    sp = row.get("spotify_resolved") or {}
    items = sp.get("items") or []
    artists: list[str] = []
    for it in items:
        for a in it.get("artists") or []:
            if a.get("name") and a["name"] not in artists:
                artists.append(a["name"])
    if not artists:
        for a in sp.get("artists") or []:
            if isinstance(a, dict) and a.get("name"):
                artists.append(a["name"])
    decade, genre = _parse_tags(row.get("raw_title") or "")
    return ToTQuery(
        id=row["id"],
        title=(row.get("title") or "").strip(),
        text=(row.get("text") or "").strip(),
        raw_title=(row.get("raw_title") or "").strip(),
        query_types=list(row.get("query_types") or []),
        split=row.get("split"),
        answer_wasabi_id=str(row["wasabi_2.0_id"]),
        answer_isrc=str(row.get("wasabi_isrc") or ""),
        answer_title=sp.get("title") or "",
        answer_artists=artists,
        answer_spotify_urls=[it["url"] for it in items if it.get("url")],
        tag_decade=decade,
        tag_genre=genre,
    )


def load(download_if_missing: bool = True) -> list[ToTQuery]:
    p = jsonl_path()
    if not p.exists():
        if not download_if_missing:
            raise FileNotFoundError(p)
        download()
    out: list[ToTQuery] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_parse_row(json.loads(line)))
    logger.info("ToT_Music : %d demandes chargées", len(out))
    return out


def answer_ids(queries: list[ToTQuery] | None = None) -> dict[str, dict]:
    """Liste des morceaux réponses, indexée par identifiant WASABI, pour la phase 1
    (`catalog/build.py` doit tous les inclure en priorité)."""
    queries = queries if queries is not None else load()
    out: dict[str, dict] = {}
    for q in queries:
        out.setdefault(q.answer_wasabi_id, {
            "wasabi_id": q.answer_wasabi_id,
            "isrc": q.answer_isrc,
            "title": q.answer_title,
            "artists": q.answer_artists,
            "n_queries": 0,
        })["n_queries"] += 1
    return out


if __name__ == "__main__":
    import collections
    logging.basicConfig(level=logging.INFO)
    qs = load()
    print(f"{len(qs)} demandes, {len(answer_ids(qs))} morceaux réponses distincts")
    print("splits :", collections.Counter(q.split for q in qs))
    print("décennies (balises) :", collections.Counter(q.tag_decade for q in qs).most_common(8))
