"""EmbeddingGemma 2 (`google/embeddinggemma-2`) en local via sentence-transformers ≥ 6.1.

Modèle ouvert (Apache 2.0, ~740 M paramètres) : texte, image, vidéo et audio dans
un même espace, 100+ langues, 768 d (MRL 512/256/128). Fiche HF lue le 28/10/2026 :
- audio **mono 16 kHz**, 25 tokens/s, ≈ 327 s max ; vidéo échantillonnée à 1 image/s
  par l'encodeur vision (la piste audio n'est pas lue) ;
- prompts fournis par le modèle : `query` = « task: search result | query: », `document` =
  « title: none | text: » ; pas de préfixe pour l'audio et la vidéo.
Même famille de préfixes que Gemini Embedding 2 (API), mais ici **rien ne quitte la machine**.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from core.audio_io import load_audio, resample, to_mono
from core.config import get_settings
from core.device import get_device
from core.embedders.base import AudioItem, Embedder

logger = logging.getLogger(__name__)

MODEL_ID = "google/embeddinggemma-2"
SR = 16_000


class GemmaEmbedder(Embedder):
    name = "gemma"
    supports_video = True
    languages = "multilingual"

    def __init__(self, model_id: str = MODEL_ID, dim: int | None = None):
        super().__init__()
        self.model_id = model_id
        self.dim = dim or 768
        self._model = None

    def version(self) -> str:
        return f"1-{self.dim}"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = str(get_device())
            logger.info("Chargement EmbeddingGemma 2 %s sur %s (dim %d)", self.model_id, device, self.dim)
            self._model = SentenceTransformer(self.model_id, device=device, truncate_dim=self.dim if self.dim != 768 else None)
            prompts = getattr(self._model, "prompts", {}) or {}
            if "query" not in prompts or "document" not in prompts:
                raise RuntimeError(f"EmbeddingGemma 2 : prompts `query`/`document` absents ({list(prompts)[:5]}) ; vérifier le modèle")

    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=SR)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, SR)

    def _encode(self, inputs, **kw) -> np.ndarray:
        return np.asarray(self._model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True,
                                             batch_size=len(inputs) or 1, show_progress_bar=False, **kw), dtype=np.float32)

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        return self._encode([{"audio": {"array": self._to_wave(it), "sampling_rate": SR}} for it in items])

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        self._load()
        return self._encode(list(texts), prompt_name="query" if task == "query" else "document")

    def _embed_video(self, paths: Sequence[str]) -> np.ndarray:
        self._load()
        return self._encode([{"video": str(p)} for p in paths])
