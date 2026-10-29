"""Fiches morceaux texte (PLAN.md phase 1.5).

Une fiche = titre, artiste, année, genre, résumé Wikipédia court (FR puis EN),
usages connus (liste manuelle), et une **description audio générée par Llama**
depuis les métadonnées (marquée `[généré]`). Pas de paroles.

- Wikipédia : recherche `list=search` puis résumé REST `page/summary` ; on
  n'accepte une page que si son titre ressemble au titre du morceau ou si sa
  description mentionne une chanson/single et l'artiste. Cache disque.
- Description audio : `core.llm.chat` en JSON (instruments probables, tempo,
  voix, ambiance, production), en anglais pour CLAP, cache par hash des
  métadonnées. Le modèle n'a que les métadonnées : la description est une
  hypothèse, jamais présentée comme un fait.

Sortie : `data/catalog/cards.parquet` (track_id, card_text, wiki_summary,
wiki_url, wiki_lang, audio_description).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from catalog.normalize import similarity
from core import llm
from core.cache import DiskCache
from core.config import get_settings

logger = logging.getLogger(__name__)

_UA = "SoundTrace/0.1 (hackathon, usage local ; contact: guillaume.beguec@gwenlake.com)"
_SONG_WORDS = re.compile(r"\b(song|single|chanson|titre|morceau|track|hit|tube)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- Wikipédia

def _mentions(artist: str, text: str) -> bool:
    """Vrai si le nom complet de l'artiste apparaît (mots entiers, sans accents/casse).
    Pour « Daft Punk feat. X » on teste la partie avant « feat ». Les noms très courts
    (« A », « M ») exigent une mention explicite du type « A (band) » / « by A »."""
    from catalog.normalize import fold
    name = re.split(r"\s+(?:feat\.?|featuring|ft\.?|&|,)\s+", artist, maxsplit=1, flags=re.IGNORECASE)[0]
    fname, ftext = fold(name), fold(text)
    if not fname:
        return False
    if len(fname) <= 2:
        raw = text.lower()
        return bool(re.search(rf"\((?:the )?{re.escape(name.lower())} song\)", raw)) or \
            bool(re.search(rf"\b{re.escape(name.lower())} \((?:band|group|singer|musician|groupe|chanteur|chanteuse)\)", raw))
    return bool(re.search(rf"\b{re.escape(fname)}\b", ftext))


@dataclass
class WikiSummary:
    lang: str
    title: str
    url: str
    description: str | None
    extract: str


class WikipediaRateLimited(httpx.TransportError):
    """HTTP 429 : on attend (Retry-After) et on réessaie."""


class Wikipedia:
    # Wikimedia demande un User-Agent identifiable et un débit raisonnable : on se limite
    # à ~4 req/s et on respecte Retry-After sur 429 (observé le 28/10 à ~5 req/s soutenues).
    MIN_INTERVAL = 0.25

    def __init__(self, langs: tuple[str, ...] = ("fr", "en"), timeout: float = 15.0):
        self.langs = langs
        self._http = httpx.Client(timeout=timeout, headers={"User-Agent": _UA}, follow_redirects=True)
        self._cache = DiskCache("wikipedia")
        self._last = 0.0

    @retry(retry=retry_if_exception_type(httpx.TransportError), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        import time
        wait = self.MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self._http.get(url, params=params)
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", "5") or 5)
            logger.warning("Wikipédia 429, pause %.0f s", retry_after)
            time.sleep(min(retry_after, 60))
            raise WikipediaRateLimited("429")
        r.raise_for_status()
        return r.json()

    def _cached(self, key_content: str, fn, **params):
        key = self._cache.key(key_content, **params)
        hit = self._cache.get(key)
        if hit is not None:
            return hit["v"]
        v = fn()
        self._cache.put(key, {"v": v})
        return v

    def search(self, lang: str, query: str, limit: int = 5) -> list[dict]:
        def fn():
            d = self._get(f"https://{lang}.wikipedia.org/w/api.php",
                          {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit})
            return (d or {}).get("query", {}).get("search", [])
        return self._cached(query, fn, lang=lang, kind="search", limit=limit)

    def summary(self, lang: str, title: str) -> dict | None:
        def fn():
            return self._get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}")
        return self._cached(title, fn, lang=lang, kind="summary")

    def song_summary(self, title: str, artist: str, max_chars: int = 500) -> WikiSummary | None:
        """Résumé de la page du morceau, FR puis EN. `None` si aucune page ne passe le contrôle."""
        for lang in self.langs:
            q = f'"{title}" "{artist}" ' + ("chanson" if lang == "fr" else "song")
            for hit in self.search(lang, q, limit=5):
                page = hit["title"]
                sim = similarity(re.sub(r"\s*\(.*\)$", "", page), title)
                if sim < 0.75:
                    continue
                s = self.summary(lang, page)
                if not s or s.get("type") == "disambiguation":
                    continue
                desc = s.get("description") or ""
                extract = s.get("extract") or ""
                looks_like_song = bool(_SONG_WORDS.search(desc)) or bool(_SONG_WORDS.search(extract[:200]))
                mentions_artist = _mentions(artist, page + " " + desc + " " + extract[:500])
                if not (looks_like_song and mentions_artist):
                    continue
                extract = re.sub(r"\s+", " ", extract).strip()
                if len(extract) > max_chars:
                    extract = extract[:max_chars].rsplit(" ", 1)[0] + "…"
                return WikiSummary(lang=lang, title=page, url=s.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                   description=desc or None, extract=extract)
        return None


# --------------------------------------------------------------------------- description audio (Llama)

AUDIO_DESC_SCHEMA = {
    "title": "audio_description",
    "type": "object",
    "required": ["description", "instruments", "tempo", "vocals", "mood"],
    "properties": {
        "description": {"type": "string", "description": "ONE sentence (max 30 words) in English on how the track sounds"},
        "instruments": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "tempo": {"type": "string", "enum": ["slow", "medium", "fast", "unknown"]},
        "vocals": {"type": "string", "enum": ["male", "female", "mixed", "instrumental", "unknown"]},
        "mood": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
}

_DESC_SYSTEM = (
    "You are a music expert. From song metadata only (title, artist, year, genre, language, short context), "
    "write a plausible description of how the track sounds, as a caption for an audio search engine: "
    "instruments, tempo, vocals, mood, production style. English only, ONE short sentence, be terse. Never quote lyrics. "
    "If you genuinely know the song, describe it; otherwise infer from the artist, genre and era. "
    "Do not mention that you are guessing."
)


def audio_description(meta: dict[str, Any], model: str | None = None) -> dict | None:
    """Description audio JSON générée une fois par Llama et mise en cache (hash des métadonnées)."""
    cache = DiskCache("llama_audio_desc")
    payload = {k: meta.get(k) for k in ("title", "artist", "year", "genre", "language", "wiki_summary")}
    key = cache.key(str(sorted(payload.items())), model=model or get_settings().llm_model, v=2)
    hit = cache.get(key)
    if hit is not None:
        return hit["v"]
    user = "Metadata:\n" + "\n".join(f"- {k}: {v}" for k, v in payload.items() if v)
    try:
        r = llm.chat([{"role": "system", "content": _DESC_SYSTEM}, {"role": "user", "content": user}],
                     json_schema=AUDIO_DESC_SCHEMA, temperature=0.2, max_tokens=160, model=model)
        out = r.json
    except llm.LLMError as e:
        logger.warning("description audio impossible pour %s — %s : %s", meta.get("title"), meta.get("artist"), e)
        out = None
    cache.put(key, {"v": out})
    return out


# --------------------------------------------------------------------------- fiche

def card_text(meta: dict[str, Any], wiki: WikiSummary | None, desc: dict | None) -> str:
    parts = [f"Titre : {meta['title']}", f"Artiste : {meta['artist']}"]
    if meta.get("album_title"):
        parts.append(f"Album : {meta['album_title']}")
    if meta.get("year"):
        parts.append(f"Année : {int(meta['year'])}")
    if meta.get("genre"):
        parts.append(f"Genre : {meta['genre']}")
    if meta.get("language"):
        parts.append(f"Langue : {meta['language']}")
    if meta.get("usage"):
        parts.append(f"Usage connu : {meta['usage']}")
    if meta.get("chart_years"):
        ys = sorted(int(y) for y in meta["chart_years"])
        parts.append("Tube : " + ", ".join(str(y) for y in ys) + (" (France)" if meta.get("country_hint") == "FR" else ""))
    if wiki:
        parts.append(f"Résumé ({wiki.lang}) : {wiki.extract}")
    if desc:
        d = desc.get("description", "")
        extra = []
        if desc.get("instruments"):
            extra.append("instruments: " + ", ".join(map(str, desc["instruments"])))
        if desc.get("tempo") and desc["tempo"] != "unknown":
            extra.append("tempo: " + str(desc["tempo"]))
        if desc.get("vocals") and desc["vocals"] != "unknown":
            extra.append("vocals: " + str(desc["vocals"]))
        if desc.get("mood"):
            extra.append("mood: " + ", ".join(map(str, desc["mood"])))
        parts.append("[généré] Description audio : " + d + (" (" + "; ".join(extra) + ")" if extra else ""))
    return "\n".join(parts)


def build_cards(catalog: pd.DataFrame, with_wiki: bool = True, with_llm: bool = True,
                out: Path | None = None, log_every: int = 200) -> pd.DataFrame:
    wiki = Wikipedia() if with_wiki else None
    rows = []
    for i, r in enumerate(catalog.itertuples(index=False), 1):
        meta = r._asdict()
        cy = meta.get("chart_years")
        meta["chart_years"] = [int(y) for y in cy] if cy is not None and len(cy) else []
        meta["year"] = int(meta["year"]) if meta.get("year") is not None and not pd.isna(meta["year"]) else None
        ws = None
        if wiki:
            try:
                ws = wiki.song_summary(meta["title"], meta["artist"])
            except httpx.HTTPError as e:
                logger.warning("wikipedia : %s — %s : %s", meta["title"], meta["artist"], e)
        if ws:
            meta["wiki_summary"] = ws.extract
        desc = audio_description(meta) if with_llm else None
        rows.append({
            "track_id": meta["track_id"],
            "card_text": card_text(meta, ws, desc),
            "wiki_summary": ws.extract if ws else None,
            "wiki_url": ws.url if ws else None,
            "wiki_lang": ws.lang if ws else None,
            "audio_description": desc.get("description") if desc else None,
            "desc_tempo": desc.get("tempo") if desc else None,
            "desc_vocals": desc.get("vocals") if desc else None,
        })
        if i % log_every == 0:
            logger.info("fiches : %d / %d (wiki %d, desc %d)", i, len(catalog),
                        sum(1 for x in rows if x["wiki_summary"]), sum(1 for x in rows if x["audio_description"]))
    df = pd.DataFrame(rows)
    if out is None:
        out = get_settings().data_dir / "catalog" / "cards.parquet"
    df.to_parquet(out, index=False)
    logger.info("fiches : %d écrites dans %s ; wiki %.1f %%, description %.1f %%", len(df), out,
                100 * df.wiki_summary.notna().mean(), 100 * df.audio_description.notna().mean())
    return df
