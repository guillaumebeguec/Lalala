"""CLAP (`laion/larger_clap_music`) via transformers, on MPS.

English only: French queries must be translated by Llama upstream
(see core/query_rewrite.py). Audio expected at 48 kHz mono; the 30 s
previews are encoded whole (the model fuses the 10 s windows).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from core.audio_io import load_audio, resample, to_mono
from core.device import get_device
from core.embedders.base import AudioItem, Embedder

logger = logging.getLogger(__name__)

MODEL_ID = "laion/larger_clap_music"
SR = 48_000


def _as_tensor(out) -> torch.Tensor:
    """transformers >= 5 returns a ModelOutput (pooler_output = projection) instead of a tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for k in ("pooler_output", "audio_embeds", "text_embeds"):
        v = getattr(out, k, None)
        if isinstance(v, torch.Tensor):
            return v
    return out[0]


class ClapEmbedder(Embedder):
    name = "clap"
    dim = 512
    supports_video = False
    languages = "en"

    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import ClapModel, ClapProcessor
            self.device = get_device()
            logger.info("Loading CLAP %s on %s", self.model_id, self.device)
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()
            self.dim = int(self._model.config.projection_dim)

    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=SR)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, SR)

    @torch.inference_mode()
    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        waves = [self._to_wave(it) for it in items]
        inputs = self._processor(audio=waves, sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _as_tensor(self._model.get_audio_features(**inputs))
        return feats.float().cpu().numpy()

    @torch.inference_mode()
    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        self._load()
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _as_tensor(self._model.get_text_features(**inputs))
        return feats.float().cpu().numpy()
