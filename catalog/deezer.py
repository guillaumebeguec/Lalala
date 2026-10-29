"""Client métadonnées de l'API Deezer publique (sans authentification).

Usages ici : résolution d'un morceau (ISRC, identifiant, recherche
`artist:"…" track:"…"`), playlists éditoriales et charts pour les tubes par
année, et l'URL de l'extrait de 30 s (`preview`). **Ce module ne télécharge
pas d'audio** : ce qui est fait de l'URL `preview` est décidé dans
`catalog/previews.py` selon les conditions d'utilisation Deezer (voir README).

Conditions (developers.deezer.com/termsofuse, consultées le 28/10/2026) :
usage non commercial, privé ; pas de contournement pour télécharger le
contenu ; les guidelines interdisent le stockage local/hors-ligne de l'audio.

Débit : limite constatée par les clients tiers ≈ 50 requêtes / 5 s. On se
place sous (10 req/s max, backoff sur les erreurs « Quota limit exceeded »).
Toutes les réponses sont mises en cache sur disque (clé = endpoint + params).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.cache import DiskCache

logger = logging.getLogger(__name__)

BASE_URL = "https://api.deezer.com"
_MIN_INTERVAL = 0.12  # s entre deux requêtes (≈ 8 req/s)


class DeezerError(RuntimeError):
    pass


class DeezerQuota(DeezerError):
    """Quota dépassé : on attend et on réessaie."""


class DeezerNotFound(DeezerError):
    pass


class DeezerClient:
    def __init__(self, cache_namespace: str = "deezer_api", timeout: float = 20.0):
        self._http = httpx.Client(base_url=BASE_URL, timeout=timeout, headers={"User-Agent": "SoundTrace/0.1 (recherche, usage local)"})
        self._cache = DiskCache(cache_namespace)
        self._lock = threading.Lock()
        self._last = 0.0
        self.n_requests = 0
        self.n_cached = 0

    # ---- bas niveau ----
    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = _MIN_INTERVAL - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    @retry(retry=retry_if_exception_type((DeezerQuota, httpx.TransportError)),
           stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
    def _fetch(self, path: str, params: dict[str, Any]) -> Any:
        self._throttle()
        r = self._http.get(path, params=params)
        self.n_requests += 1
        if r.status_code == 429:
            raise DeezerQuota(f"HTTP 429 sur {path}")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code")
            if code == 4 or "quota" in str(err.get("message", "")).lower():
                raise DeezerQuota(str(err))
            if code in (800, 404) or "no data" in str(err.get("message", "")).lower():
                raise DeezerNotFound(str(err))
            raise DeezerError(str(err))
        return data

    def get(self, path: str, use_cache: bool = True, **params: Any) -> Any:
        """GET avec cache disque. Les « not found » sont aussi mis en cache (valeur `None`)."""
        key = self._cache.key(path, **{k: str(v) for k, v in sorted(params.items())})
        if use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                self.n_cached += 1
                return hit["v"]
        try:
            data = self._fetch(path, params)
        except DeezerNotFound:
            data = None
        self._cache.put(key, {"v": data})  # enveloppe : un « not found » (None) est aussi un hit
        return data

    def paginate(self, path: str, limit: int = 100, max_items: int | None = None, **params: Any) -> Iterator[dict]:
        index = 0
        seen = 0
        while True:
            page = self.get(path, index=index, limit=limit, **params)
            if not page or not page.get("data"):
                return
            for item in page["data"]:
                yield item
                seen += 1
                if max_items and seen >= max_items:
                    return
            if "next" not in page:
                return
            index += limit

    # ---- morceaux ----
    def track(self, track_id: int) -> dict | None:
        return self.get(f"/track/{int(track_id)}")

    def track_by_isrc(self, isrc: str) -> dict | None:
        return self.get(f"/track/isrc:{isrc.strip().upper()}")

    def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        data = self.get("/search", q=query, limit=limit)
        return list((data or {}).get("data") or [])

    def search_track(self, artist: str, title: str, limit: int = 10) -> list[dict]:
        """Recherche avancée `artist:"…" track:"…"` ; repli sur la recherche libre si vide."""
        a = artist.replace('"', " ").strip()
        t = title.replace('"', " ").strip()
        res = self.search_tracks(f'artist:"{a}" track:"{t}"', limit=limit)
        if not res:
            res = self.search_tracks(f"{a} {t}", limit=limit)
        return res

    # ---- listes ----
    def chart_tracks(self, genre_id: int = 0, limit: int = 100) -> list[dict]:
        data = self.get(f"/chart/{genre_id}/tracks", limit=limit)
        return list((data or {}).get("data") or [])

    def search_playlists(self, query: str, limit: int = 10) -> list[dict]:
        data = self.get("/search/playlist", q=query, limit=limit)
        return list((data or {}).get("data") or [])

    def playlist(self, playlist_id: int) -> dict | None:
        return self.get(f"/playlist/{int(playlist_id)}")

    def playlist_tracks(self, playlist_id: int, max_items: int | None = None) -> list[dict]:
        return list(self.paginate(f"/playlist/{int(playlist_id)}/tracks", max_items=max_items))

    def artist_top(self, artist_id: int, limit: int = 50) -> list[dict]:
        return list(self.paginate(f"/artist/{int(artist_id)}/top", limit=min(limit, 100), max_items=limit))

    def genres(self) -> list[dict]:
        return list((self.get("/genre") or {}).get("data") or [])


def track_summary(t: dict) -> dict:
    """Projection stable d'un objet track Deezer (search, playlist, chart ou /track)."""
    return {
        "deezer_id": t.get("id"),
        "title": t.get("title"),
        "title_short": t.get("title_short"),
        "artist": (t.get("artist") or {}).get("name"),
        "deezer_artist_id": (t.get("artist") or {}).get("id"),
        "album_title": (t.get("album") or {}).get("title"),
        "deezer_album_id": (t.get("album") or {}).get("id"),
        "isrc": t.get("isrc"),
        "rank": t.get("rank"),
        "duration": t.get("duration"),
        "release_date": t.get("release_date"),
        "bpm": t.get("bpm"),
        "preview_url": t.get("preview") or None,
        "readable": t.get("readable"),
        "explicit": t.get("explicit_lyrics"),
    }
