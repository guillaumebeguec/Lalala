"""Interface commune des embedders (PLAN.md phase 3.1).

- `embed_audio(items)` : liste d'octets (mp3/wav) **ou** de tuples (signal float32, sr) ;
- `embed_text(texts, task="query"|"document")` ;
- `embed_video(paths)` : optionnel, `NotImplementedError` sinon.

Toutes les sorties sont des `np.ndarray` float32 de forme (n, dim), normalisées L2,
pour un produit scalaire direct dans FAISS `IndexFlatIP`. Les implémentations
locales (CLAP, PE-AV) tournent sur `core.device.get_device()` ; Gemini passe par
l'API. Toutes mettent en cache par hash du contenu (`core.cache`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

from core.cache import DiskCache, content_hash

logger = logging.getLogger(__name__)

AudioItem = bytes | tuple[np.ndarray, int]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, eps)).astype(np.float32)


class Embedder(ABC):
    name: str = "base"
    dim: int = 0
    supports_video: bool = False
    languages: str = "multilingual"  # ou "en"

    def __init__(self):
        self._cache = DiskCache(f"emb_{self.name}")

    # ---- à implémenter ----
    @abstractmethod
    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray: ...

    @abstractmethod
    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray: ...

    def _embed_video(self, paths: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} n'encode pas la vidéo")

    # ---- clés de cache ----
    def _audio_key(self, item: AudioItem) -> str:
        if isinstance(item, (bytes, bytearray)):
            return content_hash(bytes(item), embedder=self.name, kind="audio", v=self.version())
        x, sr = item
        return content_hash(np.ascontiguousarray(x, dtype=np.float32).tobytes(), embedder=self.name, kind="audio", sr=sr, v=self.version())

    def _text_key(self, text: str, task: str) -> str:
        return content_hash(text, embedder=self.name, kind="text", task=task, v=self.version())

    def version(self) -> str:
        """Change quand le modèle ou le prétraitement change (invalide le cache)."""
        return "1"

    # ---- API publique avec cache et batching ----
    def _with_cache(self, keys: list[str], compute, batch_size: int) -> np.ndarray:
        out: list[np.ndarray | None] = [self._cache.get_array(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(missing), batch_size):
            idx = missing[start : start + batch_size]
            vecs = compute(idx)
            vecs = l2_normalize(vecs)
            for j, i in enumerate(idx):
                self._cache.put_array(keys[i], vecs[j])
                out[i] = vecs[j]
        return np.stack([np.asarray(v, dtype=np.float32) for v in out]) if out else np.zeros((0, self.dim), np.float32)

    def embed_audio(self, items: Sequence[AudioItem], batch_size: int = 8) -> np.ndarray:
        items = list(items)
        keys = [self._audio_key(it) for it in items]
        return self._with_cache(keys, lambda idx: self._embed_audio([items[i] for i in idx]), batch_size)

    def embed_text(self, texts: Sequence[str], task: str = "query", batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        keys = [self._text_key(t, task) for t in texts]
        return self._with_cache(keys, lambda idx: self._embed_text([texts[i] for i in idx], task), batch_size)

    def embed_video(self, paths: Sequence[str], batch_size: int = 2) -> np.ndarray:
        paths = [str(p) for p in paths]
        from pathlib import Path
        keys = [content_hash(Path(p), embedder=self.name, kind="video", v=self.version()) for p in paths]
        return self._with_cache(keys, lambda idx: self._embed_video([paths[i] for i in idx]), batch_size)


def get_embedder(name: str) -> Embedder:
    """Fabrique par nom (`clap`, `peav`, `gemma`, `gemini`, `coverhunter`), import différé pour ne charger que le nécessaire."""
    if name == "clap":
        from core.embedders.clap import ClapEmbedder
        return ClapEmbedder()
    if name == "peav":
        from core.embedders.peav import PEAVEmbedder
        return PEAVEmbedder()
    if name == "gemma":
        from core.embedders.gemma import GemmaEmbedder
        return GemmaEmbedder()
    if name == "gemini":
        from core.embedders.gemini import GeminiEmbedder
        return GeminiEmbedder()
    if name == "coverhunter":
        from core.embedders.coverhunter import CoverHunterEmbedder
        return CoverHunterEmbedder()
    if name == "hum2song":
        from core.embedders.hum2song import Hum2SongEmbedder
        return Hum2SongEmbedder()
    raise ValueError(f"embedder inconnu : {name}")
