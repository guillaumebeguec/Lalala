"""Extraits de 30 s : résolution des métadonnées et accès à l'audio (PLAN.md phase 1.4).

Deux étapes séparées, parce que les conditions Deezer les traitent différemment :

1. **Résolution (métadonnées, toujours autorisée)** — `resolve_all` : pour chaque
   morceau du catalogue, l'objet track Deezer courant (`/track/{id}`, repli
   `/track/isrc:`), d'où l'URL d'extrait fraîche, `readable`, `rank`,
   `release_date`, `bpm`, et un contrôle titre/artiste (similarité de chaînes)
   pour détecter les mauvais appariements. Résultat : `data/catalog/previews.parquet`.
   Repli iTunes Search API (`previewUrl`) quand Deezer n'a pas d'extrait.

2. **Audio** — `fetch_audio` : ouvre l'URL d'extrait et renvoie les octets MP3.
   Le mode de conservation est réglé par `PREVIEW_STORAGE` :
     - `memory` (défaut) : rien n'est écrit sur disque ; les embeddings et
       empreintes sont calculés à la volée et seuls les vecteurs sont conservés ;
     - `disk` : cache dans `data/previews/` (jamais commité). À n'activer qu'après
       validation des conditions d'utilisation (les guidelines Deezer interdisent
       le stockage local de l'audio ; voir README).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from catalog.deezer import DeezerClient, DeezerError, track_summary
from catalog.normalize import is_version_of, norm_artist, norm_title, similarity
from core.cache import DiskCache
from core.config import get_settings

logger = logging.getLogger(__name__)

ITUNES_URL = "https://itunes.apple.com/search"
_ITUNES_MIN_INTERVAL = 3.2  # ≈ 20 requêtes / minute (limite documentée par Apple)


def previews_path() -> Path:
    return get_settings().data_dir / "catalog" / "previews.parquet"


# --------------------------------------------------------------------------- iTunes (repli)

class ITunesClient:
    def __init__(self):
        self._http = httpx.Client(timeout=20, headers={"User-Agent": "SoundTrace/0.1"})
        self._cache = DiskCache("itunes_api")
        self._last = 0.0

    @retry(retry=retry_if_exception_type(httpx.TransportError), stop=stop_after_attempt(4),
           wait=wait_exponential(min=1, max=20), reraise=True)
    def search(self, artist: str, title: str, country: str = "fr") -> list[dict]:
        key = self._cache.key(f"{artist}|{title}", country=country)
        hit = self._cache.get(key)
        if hit is not None:
            return hit["v"]
        wait = _ITUNES_MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self._http.get(ITUNES_URL, params={"term": f"{artist} {title}", "entity": "song", "limit": 10, "country": country})
        if r.status_code == 403 or r.status_code == 429:
            raise httpx.TransportError(f"iTunes {r.status_code}")
        r.raise_for_status()
        res = r.json().get("results", [])
        self._cache.put(key, {"v": res})
        return res

    def best_match(self, artist: str, title: str, min_sim: float = 0.8) -> dict | None:
        best, best_s = None, 0.0
        for c in self.search(artist, title):
            s = 0.6 * similarity(c.get("trackName"), title) + 0.4 * similarity(c.get("artistName"), artist)
            if is_version_of(c.get("trackName")) and not is_version_of(title):
                s -= 0.15
            if s > best_s and c.get("previewUrl"):
                best, best_s = c, s
        return best if best_s >= min_sim else None


# --------------------------------------------------------------------------- résolution

def _match_score(cat_title: str, cat_artist: str, t: dict) -> float:
    """Similarité titre/artiste, sur les chaînes brutes et normalisées (sans « (feat. …) », « - Remaster »)."""
    tt, ta = t.get("title"), (t.get("artist") or {}).get("name")
    st = max(similarity(cat_title, tt), similarity(norm_title(cat_title), norm_title(tt)))
    sa = max(similarity(cat_artist, ta), similarity(norm_artist(cat_artist), norm_artist(ta)))
    return 0.6 * st + 0.4 * sa


def resolve_one(dz: DeezerClient, row, min_sim: float = 0.7) -> dict:
    """Retourne la ligne previews pour un morceau du catalogue (sans audio)."""
    out = {"track_id": row.track_id, "deezer_id": None, "preview_url": None, "provider": None, "status": "none",
           "match_score": None, "resolved_title": None, "resolved_artist": None, "release_date": None,
           "bpm": None, "rank": None, "duration": None}
    t = None
    try:
        if pd.notna(row.deezer_id):
            t = dz.track(int(row.deezer_id))
            if t and (not t.get("readable", True) or not t.get("preview")):
                t = None
        if t is None and isinstance(row.isrc, str) and row.isrc:
            t = dz.track_by_isrc(row.isrc)
            if t and (not t.get("readable", True) or not t.get("preview")):
                t = None
        if t is None:
            for c in dz.search_track(row.artist, row.title, limit=5):
                if c.get("preview") and c.get("readable", True) and _match_score(row.title, row.artist, c) >= 0.85:
                    t = c
                    break
    except DeezerError as e:
        logger.warning("deezer %s : %s", row.track_id, e)
    if t:
        s = track_summary(t)
        score = _match_score(row.title, row.artist, t)
        out.update(deezer_id=s["deezer_id"], preview_url=s["preview_url"], provider="deezer",
                   status="ok" if score >= min_sim else "mismatch", match_score=round(score, 3),
                   resolved_title=s["title"], resolved_artist=s["artist"], release_date=s["release_date"],
                   bpm=s["bpm"], rank=s["rank"], duration=s["duration"])
    return out


def resolve_all(catalog: pd.DataFrame, use_itunes: bool = True, workers: int = 4, log_every: int = 500,
                out: Path | None = None) -> pd.DataFrame:
    """Résout tous les morceaux ; reprise : les lignes déjà présentes dans previews.parquet sont gardées."""
    out = out or previews_path()
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_ids = set(done.track_id) if len(done) else set()
    todo = catalog[~catalog.track_id.isin(done_ids)]
    logger.info("résolution : %d à faire, %d déjà faits", len(todo), len(done_ids))
    dz = DeezerClient()
    rows: list[dict] = list(done.to_dict("records")) if len(done) else []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(lambda row: resolve_one(dz, row), todo.itertuples(index=False)), 1):
            rows.append(r)
            if i % log_every == 0:
                df = pd.DataFrame(rows)
                df.to_parquet(out, index=False)
                rate = i / (time.time() - t0)
                logger.info("résolution : %d / %d (%.1f/s, ok %.1f %%, reste ≈ %.0f min)", i, len(todo), rate,
                            100 * df.status.eq("ok").mean(), (len(todo) - i) / max(rate, 1e-6) / 60)
    df = pd.DataFrame(rows)
    if use_itunes:
        missing = df[df.status != "ok"]
        if len(missing):
            logger.info("repli iTunes pour %d morceaux (≈ %.0f min à 20 req/min)", len(missing), len(missing) * _ITUNES_MIN_INTERVAL / 60)
            it = ITunesClient()
            cat = catalog.set_index("track_id")
            for idx in missing.index:
                row = cat.loc[df.at[idx, "track_id"]]
                try:
                    m = it.best_match(row.artist, row.title)
                except httpx.HTTPError as e:
                    logger.warning("itunes %s : %s", row.name, e)
                    continue
                if m:
                    df.loc[idx, ["preview_url", "provider", "status", "resolved_title", "resolved_artist", "match_score"]] = [
                        m["previewUrl"], "itunes", "ok", m.get("trackName"), m.get("artistName"),
                        round(0.6 * similarity(row.title, m.get("trackName")) + 0.4 * similarity(row.artist, m.get("artistName")), 3)]
    df.to_parquet(out, index=False)
    logger.info("résolution terminée : %d lignes, ok %.1f %% (deezer %d, itunes %d), mismatch %d, none %d ; requêtes Deezer %d, cache %d",
                len(df), 100 * df.status.eq("ok").mean(), int((df.provider == "deezer").sum()), int((df.provider == "itunes").sum()),
                int(df.status.eq("mismatch").sum()), int(df.status.eq("none").sum()), dz.n_requests, dz.n_cached)
    return df


def load_previews() -> pd.DataFrame:
    return pd.read_parquet(previews_path())


# --------------------------------------------------------------------------- audio

def storage_mode() -> str:
    import os
    return os.environ.get("PREVIEW_STORAGE", "memory").strip().lower()


_audio_http = httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "SoundTrace/0.1"})
_fresh: dict[str, tuple[float, str | None]] = {}
_FRESH_TTL = 20 * 60  # les URL d'extrait Deezer sont signées (`hdnea`) et expirent en moins d'une heure
_dz_client = None


def fresh_preview_url(track_id: str, fallback: str | None = None) -> str | None:
    """URL d'extrait courante : relue via `/track/{id}` (sans cache) si l'identifiant est Deezer ;
    mémorisée 20 min. Pour iTunes (URL stables), on renvoie `fallback`."""
    global _dz_client
    now = time.time()
    hit = _fresh.get(track_id)
    if hit and now - hit[0] < _FRESH_TTL:
        return hit[1] or fallback
    if not track_id.startswith("dz"):
        return fallback
    try:
        if _dz_client is None:
            _dz_client = DeezerClient()
        t = _dz_client.get(f"/track/{int(track_id[2:])}", use_cache=False)
        url = (t or {}).get("preview") or None
    except (DeezerError, ValueError) as e:
        logger.warning("rafraîchissement d'extrait %s : %s", track_id, e)
        url = None
    _fresh[track_id] = (now, url)
    return url or fallback


@retry(retry=retry_if_exception_type(httpx.TransportError), stop=stop_after_attempt(4),
       wait=wait_exponential(min=1, max=15), reraise=True)
def fetch_audio(track_id: str, preview_url: str) -> bytes | None:
    """Octets MP3 de l'extrait. En mode `disk`, cache dans data/previews/<track_id>.mp3.
    Sur 403/404 (URL signée expirée), l'URL est rafraîchie une fois via l'API Deezer."""
    mode = storage_mode()
    path = get_settings().previews_dir / f"{track_id}.mp3"
    if mode == "disk" and path.exists():
        return path.read_bytes()
    hit = _fresh.get(track_id)
    url = hit[1] if hit and time.time() - hit[0] < _FRESH_TTL and hit[1] else preview_url
    r = _audio_http.get(url)
    if r.status_code in (403, 404, 410) and track_id.startswith("dz"):
        url2 = fresh_preview_url(track_id)
        if url2 and url2 != url:
            r = _audio_http.get(url2)
    if r.status_code in (403, 404, 410):
        logger.warning("extrait indisponible (%s) : %s", r.status_code, track_id)
        return None
    r.raise_for_status()
    data = r.content
    if mode == "disk":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def iter_audio(previews: pd.DataFrame, track_ids: list[str] | None = None) -> Iterator[tuple[str, bytes]]:
    """Itère (track_id, mp3) sur les extraits disponibles, dans l'ordre du DataFrame."""
    df = previews[previews.status == "ok"]
    if track_ids is not None:
        df = df[df.track_id.isin(track_ids)]
    for r in df.itertuples(index=False):
        try:
            data = fetch_audio(r.track_id, r.preview_url)
        except httpx.HTTPError as e:
            logger.warning("extrait %s : %s", r.track_id, e)
            continue
        if data:
            yield r.track_id, data
