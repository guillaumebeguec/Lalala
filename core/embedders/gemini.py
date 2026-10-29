"""Gemini Embedding 2 (`gemini-embedding-2`) via le SDK google-genai.

Doc officielle (ai.google.dev/gemini-api/docs/embeddings, lue le 28/10/2026) :
- entrées texte, image, vidéo, audio, PDF dans un même espace ; 8 192 tokens max ;
- audio : 180 s max, MP3 ou WAV, en octets inline (`types.Part.from_bytes`) ;
- vidéo : 120 s max, MP4/MOV, 32 images max, **la piste audio n'est pas traitée** ;
- pas de `task_type` pour ce modèle : l'instruction est un préfixe texte
  (`task: search result | query: …` pour une requête, `title: … | text: …` pour un document) ;
- `output_dimensionality` 128–3072 (768 / 1536 / 3072 recommandés), renormalisé par le modèle ;
- API batch à moitié prix pour le catalogue (à utiliser pour l'indexation complète).

Tout ce qui est envoyé ici part chez Google (voir README). Désactivé si
USE_GEMINI=false ou sans clé : `get_embedder("gemini")` lève alors une erreur claire.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.audio_io import load_audio, to_wav_bytes
from core.config import get_settings
from core.embedders.base import AudioItem, Embedder

logger = logging.getLogger(__name__)

QUERY_PREFIX = "task: search result | query: "


class GeminiDisabled(RuntimeError):
    pass


def _is_retryable(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("429", "resource_exhausted", "503", "unavailable", "deadline", "500", "internal"))


class GeminiEmbedder(Embedder):
    name = "gemini"
    supports_video = True
    languages = "multilingual"

    def __init__(self):
        super().__init__()
        s = get_settings()
        if not s.use_gemini:
            raise GeminiDisabled("USE_GEMINI=false")
        if not s.gemini_api_key:
            raise GeminiDisabled("GEMINI_API_KEY absente")
        from google import genai
        self._client = genai.Client(api_key=s.gemini_api_key)
        self.model = s.gemini_model
        self.dim = s.gemini_dim
        self.n_calls = 0

    def version(self) -> str:
        return f"1-{self.model}-{self.dim}"

    @retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _call(self, contents) -> np.ndarray:
        from google.genai import types
        try:
            r = self._client.models.embed_content(model=self.model, contents=contents,
                                                  config=types.EmbedContentConfig(output_dimensionality=self.dim))
        except Exception as e:  # noqa: BLE001
            if _is_retryable(e):
                logger.warning("Gemini : %s, nouvel essai", str(e)[:120])
                raise
            raise RuntimeError(f"Gemini : {e}") from e
        self.n_calls += 1
        return np.asarray([e.values for e in r.embeddings], dtype=np.float32)

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        prefixed = [QUERY_PREFIX + t if task == "query" else t for t in texts]
        # Plusieurs textes par requête sont acceptés (liste de contents).
        return self._call(prefixed)

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        from google.genai import types
        vecs = []
        for it in items:
            if isinstance(it, (bytes, bytearray)):
                data, mime = bytes(it), ("audio/mpeg" if bytes(it[:3]) in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") else "audio/wav")
            else:
                x, sr = it
                data, mime = to_wav_bytes(x, sr), "audio/wav"
            vecs.append(self._call([types.Part.from_bytes(data=data, mime_type=mime)])[0])
        return np.stack(vecs)

    def _embed_video(self, paths: Sequence[str]) -> np.ndarray:
        from google.genai import types
        from pathlib import Path
        vecs = []
        for p in paths:
            mime = "video/quicktime" if str(p).lower().endswith(".mov") else "video/mp4"
            vecs.append(self._call([types.Part.from_bytes(data=Path(p).read_bytes(), mime_type=mime)])[0])
        return np.stack(vecs)
