"""Index de recherche : FAISS (`IndexFlatIP`, vecteurs normalisés) par (embedder × modalité)
et BM25 sur les fiches (PLAN.md phase 3.5).

Chaque index vit dans data/index/<embedder>_<modalité>/ : `vectors.npy`, `ids.json`.
API : `search(query_vec, embedder, modality, k)` → [(track_id, score)…] et
`bm25_search(text, k)`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from core.config import get_settings
from core.embedders.base import l2_normalize

logger = logging.getLogger(__name__)

Hit = tuple[str, float]


def index_dir(embedder: str, modality: str) -> Path:
    d = get_settings().index_dir / f"{embedder}_{modality}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class VectorIndex:
    embedder: str
    modality: str          # "audio" (extraits) ou "text" (fiches)
    ids: list[str]
    index: faiss.Index

    @classmethod
    def build(cls, embedder: str, modality: str, ids: list[str], vectors: np.ndarray) -> "VectorIndex":
        vectors = l2_normalize(vectors)
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(vectors)
        return cls(embedder, modality, list(ids), idx)

    def save(self) -> Path:
        d = index_dir(self.embedder, self.modality)
        np.save(d / "vectors.npy", self.index.reconstruct_n(0, self.index.ntotal))
        (d / "ids.json").write_text(json.dumps(self.ids))
        return d

    @classmethod
    def load(cls, embedder: str, modality: str) -> "VectorIndex":
        d = index_dir(embedder, modality)
        vectors = np.load(d / "vectors.npy")
        ids = json.loads((d / "ids.json").read_text())
        return cls.build(embedder, modality, ids, vectors)

    @classmethod
    def exists(cls, embedder: str, modality: str) -> bool:
        d = get_settings().index_dir / f"{embedder}_{modality}"
        return (d / "vectors.npy").exists() and (d / "ids.json").exists()

    def search(self, query: np.ndarray, k: int = 20) -> list[list[Hit]]:
        q = l2_normalize(query)
        scores, idx = self.index.search(q, min(k, self.index.ntotal))
        return [[(self.ids[j], float(s)) for j, s in zip(row_i, row_s) if j >= 0] for row_i, row_s in zip(idx, scores)]

    def vector_of(self, track_id: str) -> np.ndarray | None:
        try:
            return self.index.reconstruct(self.ids.index(track_id))
        except ValueError:
            return None


# --------------------------------------------------------------------------- BM25

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of and or to in on at for with by from is it this that de la le les des du un une et en au aux
pour par sur dans est ce cette ces qui que quoi dont où ne pas plus très il elle ils elles je tu nous vous se sa son ses
leur leurs mon ma mes ton ta tes notre votre y""".split())


def tokenize(text: str) -> list[str]:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in _TOKEN_RE.findall(s) if t not in _STOP and len(t) > 1]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, text: str, k: int = 20) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(text))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path | None = None) -> Path:
        import pickle
        path = path or (get_settings().index_dir / "bm25_cards.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        import pickle
        path = path or (get_settings().index_dir / "bm25_cards.pkl")
        return pickle.loads(path.read_bytes())
