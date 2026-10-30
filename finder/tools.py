"""Outils de l'agent (PLAN.md phase 6.1) : recherche catalogue hybride, appariement audio,
recherche web, question à l'utilisateur, proposition.

`Toolbox` charge paresseusement index et embedders d'un sous-ensemble (`dev` / `full`).
Chaque outil renvoie des données sérialisables (pour Llama) et des candidats
`Candidate` (pour l'état de l'agent et le front).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core import query_rewrite
from core.config import get_settings
from core.fingerprint import FingerprintIndex, fpcalc_raw
from core.fusion import rrf, soft_bonus
from core.index import BM25Index, VectorIndex
from core.rerank import rerank


def resample_to16(x: np.ndarray, sr: int) -> np.ndarray:
    from core.audio_io import resample, to_mono
    return resample(to_mono(np.asarray(x)), sr, 16_000)

logger = logging.getLogger(__name__)

FINGERPRINT_EXACT = 0.85


@dataclass
class Candidate:
    track_id: str
    title: str
    artist: str
    year: int | None
    score: float
    source: str                      # catalog | web | fingerprint
    reason: str = ""
    preview_url: str | None = None
    in_catalog: bool = True
    search_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "titre": self.title, "artiste": self.artist, "annee": self.year,
                "score": round(float(self.score), 4), "source": self.source, "raison": self.reason,
                "preview_url": self.preview_url, "in_catalog": self.in_catalog, "search_url": self.search_url}


@dataclass
class Toolbox:
    subset: str = "dev"
    embedders: list[str] | None = None
    use_rewrite: bool = True
    use_rerank: bool = field(default_factory=lambda: __import__("os").environ.get("SOUNDTRACE_RERANK", "1") != "0")
    rerank_alpha: float = 0.5
    rerank_top: int = 10            # candidats notés par Llama (2 lots de 5 : ≈ 30 s sur M2) ; 20 pour l'évaluation
    rrf_k: int = 60
    top_k: int = 20
    _catalog: pd.DataFrame | None = None
    _cards: dict[str, str] | None = None
    _bm25: BM25Index | None = None
    _vec: dict[tuple[str, str], VectorIndex] = field(default_factory=dict)
    _emb: dict[str, Any] = field(default_factory=dict)
    _fp: FingerprintIndex | None = None
    _previews: pd.DataFrame | None = None
    _web: Any = None
    queries_done: list[str] = field(default_factory=list)

    # ---- chargement paresseux ----
    @property
    def catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            from catalog import build as cb
            self._catalog = cb.load(self.subset).set_index("track_id", drop=False)
        return self._catalog

    @property
    def cards(self) -> dict[str, str]:
        if self._cards is None:
            p = get_settings().data_dir / "catalog" / f"cards_{self.subset}.parquet"
            df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["track_id", "card_text"])
            self._cards = dict(zip(df.track_id, df.card_text))
        return self._cards

    @property
    def previews(self) -> pd.DataFrame:
        if self._previews is None:
            from catalog.previews import previews_path
            p = previews_path()
            self._previews = pd.read_parquet(p).set_index("track_id") if p.exists() else pd.DataFrame()
        return self._previews

    def bm25(self) -> BM25Index | None:
        if self._bm25 is None:
            p = get_settings().index_dir / f"bm25_cards_{self.subset}.pkl"
            self._bm25 = BM25Index.load(p) if p.exists() else None
        return self._bm25

    def active_embedders(self) -> list[str]:
        return self.embedders if self.embedders is not None else get_settings().active_embedders

    def vec(self, embedder: str, modality: str) -> VectorIndex | None:
        """Index FAISS (embedder × modalité), chargé à la demande. Un index absent est re-testé à
        chaque appel (bon marché) pour que l'API en cours d'exécution voie les index construits
        après son démarrage ; un index mis à jour sur disque est rechargé (mtime)."""
        key = (embedder, modality)
        d = get_settings().index_dir / f"{embedder}_{modality}"
        p = d / "vectors.npy"
        if not p.exists():
            self._vec.pop(key, None)
            return None
        mtime = p.stat().st_mtime
        cached = self._vec.get(key)
        if cached is None or getattr(cached, "_mtime", None) != mtime:
            vi = VectorIndex.load(embedder, modality)
            vi._mtime = mtime
            self._vec[key] = vi
        return self._vec[key]

    def emb(self, name: str):
        if name not in self._emb:
            from core.embedders.base import get_embedder
            if name == "hum2song":   # côté requête : contour rogné à 20 s (fredonnement)
                from core.embedders.hum2song import Hum2SongEmbedder
                self._emb[name] = Hum2SongEmbedder(role="query")
            else:
                self._emb[name] = get_embedder(name)
        return self._emb[name]

    def is_humming(self, audio: bytes | tuple[np.ndarray, int]) -> bool:
        """Heuristique « voix seule monophonique » (C4.3) : part de trames voisées à forte saillance
        RMVPE élevée et spectre pauvre en hautes fréquences. Vrai → entrée `humming`, poids hum2song ×3."""
        try:
            from core.audio_io import load_audio
            x, sr = (load_audio(bytes(audio), sr=16_000) if isinstance(audio, (bytes, bytearray)) else
                     (resample_to16(audio[0], audio[1]), 16_000))
            h = self.emb("hum2song")
            h._load()
            track = h._trackers.rmvpe_track(h._rmvpe, x.astype(np.float32))
            sal = track[1]
            voiced = float((sal > 0.5).mean())
            spec = np.abs(np.fft.rfft(x[: sr * 10].astype(np.float64)))
            freqs = np.fft.rfftfreq(min(len(x), sr * 10), 1 / sr)
            hf = float(spec[freqs > 2500].sum() / (spec.sum() + 1e-9))
            return voiced > 0.35 and hf < 0.25
        except Exception as e:  # noqa: BLE001
            logger.debug("détection fredonnement impossible : %s", e)
            return False

    def fingerprints(self) -> FingerprintIndex | None:
        if self._fp is None and FingerprintIndex.path().exists():
            self._fp = FingerprintIndex.load()
        return self._fp

    # ---- candidats ----
    def _candidate(self, track_id: str, score: float, source: str, reason: str = "") -> Candidate | None:
        if track_id not in self.catalog.index:
            return None
        r = self.catalog.loc[track_id]
        prev = self.previews.at[track_id, "preview_url"] if len(self.previews) and track_id in self.previews.index else None
        return Candidate(track_id=track_id, title=str(r.title), artist=str(r.artist),
                         year=int(r.year) if pd.notna(r.year) else None, score=float(score), source=source,
                         reason=reason, preview_url=prev if isinstance(prev, str) else None)

    def meta_for_belief(self, track_id: str) -> dict:
        """Métadonnées lues par le modèle d'observation (phase 7) : année, langue, genre, bpm,
        usage, et les champs générés des fiches (tempo, voix) quand ils existent."""
        if track_id not in self.catalog.index:
            return {}
        r = self.catalog.loc[track_id]
        out = {"year": int(r.year) if pd.notna(r.year) else None, "language": r.language if isinstance(r.language, str) else None,
               "genre": r.genre if isinstance(r.genre, str) else None, "usage": r.usage if isinstance(r.usage, str) else None,
               "bpm": None}
        if len(self.previews) and track_id in self.previews.index:
            bpm = self.previews.at[track_id, "bpm"]
            out["bpm"] = float(bpm) if bpm is not None and not pd.isna(bpm) and bpm > 0 else None
        extra = self._card_fields().get(track_id, {})
        out.update({k: v for k, v in extra.items() if v})
        return out

    def _card_fields(self) -> dict[str, dict]:
        if not hasattr(self, "_card_fields_cache"):
            p = get_settings().data_dir / "catalog" / f"cards_{self.subset}.parquet"
            cache: dict[str, dict] = {}
            if p.exists():
                df = pd.read_parquet(p)
                for r in df.itertuples(index=False):
                    cache[r.track_id] = {"tempo": getattr(r, "desc_tempo", None), "vocals": getattr(r, "desc_vocals", None)}
            self._card_fields_cache = cache
        return self._card_fields_cache

    def meta_for_bonus(self, ids: list[str]) -> dict[str, dict]:
        out = {}
        for i in ids:
            if i in self.catalog.index:
                r = self.catalog.loc[i]
                out[i] = {"year": int(r.year) if pd.notna(r.year) else None, "language": r.language}
        return out

    # ---- outils ----
    def rewrite_query(self, text: str):
        """Étape 1 : réécriture Llama (légendes, indices, requête) ; None si indisponible."""
        if not self.use_rewrite:
            return None
        try:
            return query_rewrite.rewrite(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("réécriture impossible : %s", e)
            return None

    def retrieve(self, text: str, rw=None, hints: dict | None = None) -> dict[str, Any]:
        """Étape 2 : BM25 (brut + réécrit) + texte→fiches + légendes→audio, RRF, filtres doux.
        Renvoie {fused: [(id, score)], lists: {nom: n}, hints}."""
        lists: list[list[tuple[str, float]]] = []
        names: list[str] = []
        bm = self.bm25()
        if bm is not None:
            lists.append(bm.search(text, k=50)); names.append("bm25:brut")
            if rw and rw.query:
                lists.append(bm.search(rw.query + (" " + rw.query_fr if rw.query_fr else ""), k=50)); names.append("bm25:réécrit")
        for name in self.active_embedders():
            vt, va = self.vec(name, "text"), self.vec(name, "audio")
            if vt is None and va is None:
                continue
            emb = self.emb(name)
            english_only = getattr(emb, "languages", "") == "en"
            if getattr(emb, "languages", "") == "none":
                continue
            q_text = rw.query if (english_only and rw and rw.query) else text
            if vt is not None:
                lists.append(vt.search(emb.embed_text([q_text], task="query"), k=50)[0]); names.append(f"{name}:texte→fiches")
            if va is not None:
                caps = (rw.captions if rw and rw.captions else []) or ([rw.query] if rw and rw.query else [])
                if not english_only:
                    caps = caps + [text]
                if caps:
                    V = emb.embed_text(caps, task="query")
                    lists.append(va.search(np.mean(V, axis=0, keepdims=True), k=50)[0]); names.append(f"{name}:légendes→audio")
        fused = rrf(lists, k=self.rrf_k, top=max(self.top_k * 2, 40)) if lists else []
        h = dict(hints or {})
        if rw:
            h.setdefault("decade", rw.decade)
            h.setdefault("language", rw.language)
        if fused and (h.get("decade") or h.get("language")):
            fused = soft_bonus(fused, self.meta_for_bonus([i for i, _ in fused]), h)
        return {"fused": fused[: self.top_k], "lists": dict(zip(names, [len(l) for l in lists])), "hints": h}

    def rerank_candidates(self, text: str, fused: list[tuple[str, float]], hints: dict | None = None) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """Étape 3 : reranking Llama du top `rerank_top` ; les autres gardent l'ordre de fusion."""
        if not (self.use_rerank and fused and self.cards):
            return fused, {}
        h = {k: v for k, v in (hints or {}).items() if v}
        head, tail = fused[: self.rerank_top], fused[self.rerank_top :]
        rr = rerank(text + ("\nClues: " + str(h) if h else ""), head, self.cards, alpha=self.rerank_alpha, top=self.rerank_top)
        floor = min((r.final for r in rr), default=0.0)
        out = [(r.id, r.final) for r in rr] + [(i, floor * 0.9 * (1 - k / max(len(tail), 1))) for k, (i, _) in enumerate(tail)]
        return out, {r.id: r.reason for r in rr}

    def to_candidates(self, fused: list[tuple[str, float]], reasons: dict[str, str] | None = None, source: str = "catalog") -> list[Candidate]:
        reasons = reasons or {}
        return [c for c in (self._candidate(i, s, source, reasons.get(i, "")) for i, s in fused) if c]

    def catalog_search(self, text: str, hints: dict | None = None) -> dict[str, Any]:
        """Recherche hybride complète (réécriture → recherche → reranking). Renvoie {candidates, rewrite, lists}."""
        self.queries_done.append(text)
        rw = self.rewrite_query(text)
        ret = self.retrieve(text, rw, hints)
        fused, reasons = self.rerank_candidates(text, ret["fused"], ret["hints"])
        return {"candidates": self.to_candidates(fused, reasons), "rewrite": rw, "lists": ret["lists"]}

    def fingerprint_match(self, audio: bytes | tuple[np.ndarray, int]) -> Candidate | None:
        """Étape empreinte : Chromaprint ; un candidat `fingerprint` si le même enregistrement est trouvé."""
        fp = self.fingerprints()
        if fp is None:
            return None
        try:
            hits = fp.search(fpcalc_raw(audio), k=3)
            if hits and hits[0][1] >= FINGERPRINT_EXACT:
                return self._candidate(hits[0][0], hits[0][1], "fingerprint", "empreinte audio identique (même enregistrement)")
        except Exception as e:  # noqa: BLE001
            logger.warning("chromaprint : %s", e)
        return None

    def audio_similarity(self, audio: bytes | tuple[np.ndarray, int], exact: Candidate | None = None,
                         embedders: list[str] | None = None, humming: bool = False) -> dict[str, Any]:
        """Étape similarité : audio → audio pour les embedders demandés (par défaut tous les actifs),
        fusion RRF pondérée (CoverHunter ×2 sans empreinte exacte ; hum2song ×3 si fredonnement,
        généralistes ×0,5 dans ce cas). Renvoie {candidates, lists}."""
        lists, weights, names = [], [], []
        for name in (embedders or self.active_embedders()):
            va = self.vec(name, "audio")
            if va is None:
                continue
            lists.append(va.search(self.emb(name).embed_audio([audio]), k=50)[0])
            if name == "coverhunter":
                w = 2.0 if exact is None else 1.0
            elif name == "hum2song":
                w = 3.0 if humming else 0.5
            else:
                w = 0.5 if humming else 1.0
            weights.append(w)
            names.append(name)
        fused = rrf(lists, k=self.rrf_k, weights=weights, top=self.top_k) if lists else []
        cover_top = {i for i, _ in lists[names.index("coverhunter")][:5]} if "coverhunter" in names else set()
        hum_top = {i for i, _ in lists[names.index("hum2song")][:5]} if "hum2song" in names else set()
        cands = []
        for i, sc in fused:
            c = self._candidate(i, sc, "catalog")
            if c:
                if humming and i in hum_top:
                    c.reason = "Your hummed melody matches the vocal line."
                    c.source = "melody"
                elif i in cover_top and exact is None:
                    c.reason = "Same song, different version (cover or live)."
                    c.source = "version"
                cands.append(c)
        return {"candidates": cands, "lists": dict(zip(names, [len(l) for l in lists]))}

    def audio_match(self, audio: bytes | tuple[np.ndarray, int]) -> dict[str, Any]:
        """Empreinte puis similarité (agent conversationnel et évaluation)."""
        exact = self.fingerprint_match(audio)
        res = self.audio_similarity(audio, exact)
        cands = res["candidates"]
        if exact:
            cands = [exact] + [c for c in cands if c.track_id != exact.track_id]
        return {"candidates": cands, "exact": exact, "lists": res["lists"]}

    def web_search(self, query: str, k: int = 6) -> list[dict]:
        self.queries_done.append("web:" + query)
        if self._web is None:
            from core.web_search import WebSearch, WebSearchDisabled
            try:
                self._web = WebSearch()
            except WebSearchDisabled as e:
                logger.info("recherche web désactivée : %s", e)
                self._web = False
        if not self._web:
            return []
        return self._web.search(query, k=k)

    def lookup_in_catalog(self, title: str, artist: str) -> Candidate | None:
        """Pour un titre trouvé sur le web : le retrouver dans le catalogue (sinon candidat « hors catalogue »)."""
        from catalog.normalize import dedup_key, norm_artist, norm_title, similarity
        key = dedup_key(title, artist)
        df = self.catalog
        if "_key" not in df.columns:
            df["_key"] = [dedup_key(t, a) for t, a in zip(df.title, df.artist)]
        hit = df[df["_key"] == key]
        if len(hit):
            return self._candidate(hit.iloc[0].track_id, 1.0, "web", "trouvé sur le web, présent dans le catalogue")
        na = norm_artist(artist)
        cand = df[df.artist.map(norm_artist) == na]
        best, best_s = None, 0.0
        for r in cand.itertuples(index=False):
            s = similarity(norm_title(r.title), norm_title(title))
            if s > best_s:
                best, best_s = r.track_id, s
        if best and best_s >= 0.85:
            return self._candidate(best, best_s, "web", "trouvé sur le web, présent dans le catalogue")
        q = f"{title} {artist}".strip()
        return Candidate(track_id=f"web:{key}", title=title, artist=artist, year=None, score=0.5, source="web",
                         reason="trouvé sur le web, absent du catalogue (pas d'extrait)", in_catalog=False,
                         search_url="https://www.deezer.com/search/" + q.replace(" ", "%20"))
