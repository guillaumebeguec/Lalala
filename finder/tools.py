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
    use_rerank: bool = True
    rerank_alpha: float = 0.5
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
        key = (embedder, modality)
        if key not in self._vec:
            self._vec[key] = VectorIndex.load(embedder, modality) if VectorIndex.exists(embedder, modality) else None
        return self._vec[key]

    def emb(self, name: str):
        if name not in self._emb:
            from core.embedders.base import get_embedder
            self._emb[name] = get_embedder(name)
        return self._emb[name]

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
    def catalog_search(self, text: str, hints: dict | None = None) -> dict[str, Any]:
        """Recherche hybride : BM25 (requête brute + réécrite) + texte→fiches + texte→audio (légendes), RRF,
        filtres doux, puis reranking Llama. Renvoie {candidates, rewrite, lists}."""
        self.queries_done.append(text)
        lists: list[list[tuple[str, float]]] = []
        names: list[str] = []
        rw = None
        if self.use_rewrite:
            try:
                rw = query_rewrite.rewrite(text)
            except Exception as e:  # noqa: BLE001
                logger.warning("réécriture impossible : %s", e)
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
            q_text = text
            if english_only and rw and rw.query:
                q_text = rw.query
            if vt is not None:
                lists.append(vt.search(emb.embed_text([q_text], task="query"), k=50)[0]); names.append(f"{name}:texte→fiches")
            if va is not None:
                caps = (rw.captions if rw and rw.captions else []) or ([rw.query] if rw and rw.query else [])
                if not english_only:
                    caps = caps + [text]
                if caps:
                    V = emb.embed_text(caps, task="query")
                    merged = va.search(np.mean(V, axis=0, keepdims=True), k=50)[0]
                    lists.append(merged); names.append(f"{name}:légendes→audio")
        fused = rrf(lists, k=self.rrf_k, top=max(self.top_k * 2, 40)) if lists else []
        h = dict(hints or {})
        if rw:
            h.setdefault("decade", rw.decade)
            h.setdefault("language", rw.language)
        if fused and (h.get("decade") or h.get("language")):
            fused = soft_bonus(fused, self.meta_for_bonus([i for i, _ in fused]), h)
        fused = fused[: self.top_k]
        reasons: dict[str, str] = {}
        if self.use_rerank and fused and self.cards:
            rr = rerank(text + ("\nClues: " + str({k: v for k, v in h.items() if v}) if h else ""), fused, self.cards, alpha=self.rerank_alpha)
            fused = [(r.id, r.final) for r in rr]
            reasons = {r.id: r.reason for r in rr}
        cands = [c for c in (self._candidate(i, s, "catalog", reasons.get(i, "")) for i, s in fused) if c]
        return {"candidates": cands, "rewrite": rw, "lists": dict(zip(names, [len(l) for l in lists]))}

    def audio_match(self, audio: bytes | tuple[np.ndarray, int]) -> dict[str, Any]:
        """Embedding audio → audio (fusion des embedders) + Chromaprint en parallèle."""
        exact = None
        fp = self.fingerprints()
        if fp is not None:
            try:
                hits = fp.search(fpcalc_raw(audio), k=3)
                if hits and hits[0][1] >= FINGERPRINT_EXACT:
                    exact = self._candidate(hits[0][0], hits[0][1], "fingerprint", "empreinte audio identique (même enregistrement)")
            except Exception as e:  # noqa: BLE001
                logger.warning("chromaprint : %s", e)
        lists = []
        for name in self.active_embedders():
            va = self.vec(name, "audio")
            if va is None:
                continue
            lists.append(va.search(self.emb(name).embed_audio([audio]), k=50)[0])
        fused = rrf(lists, k=self.rrf_k, top=self.top_k) if lists else []
        cands = [c for c in (self._candidate(i, s, "catalog") for i, s in fused) if c]
        if exact:
            cands = [exact] + [c for c in cands if c.track_id != exact.track_id]
        return {"candidates": cands, "exact": exact}

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
