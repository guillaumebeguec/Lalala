"""PE-AV (Perception Encoder Audio-Visual, Meta) via transformers ≥ 5.21, sur MPS.

Poids : `facebook/pe-av-large` (Apache 2.0, non gatés d'après la fiche HF). API
lue dans la fiche du modèle et la signature du processeur :
    inputs = processor(audio=[np.ndarray…], text=[…], videos=[chemins…], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    outputs.audio_embeds / outputs.text_audio_embeds (texte aligné à l'audio)
    outputs.video_embeds / outputs.text_video_embeds (texte aligné à la vidéo)
La fréquence d'échantillonnage attendue est lue dans `processor.feature_extractor.sampling_rate`.
Si l'API diffère au chargement, on s'arrête avec une erreur explicite (pas de devinette).
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

MODEL_ID = "facebook/pe-av-large"


class PEAVEmbedder(Embedder):
    name = "peav"
    dim = 0
    supports_video = True
    languages = "en"

    def __init__(self, model_id: str = MODEL_ID, text_space: str = "audio"):
        super().__init__()
        self.model_id = model_id
        self.text_space = text_space  # "audio" : texte aligné à l'audio ; "video" : aligné à la vidéo
        self._model = None
        self._processor = None
        self.sr = None

    def version(self) -> str:
        return f"1-{self.text_space}"

    def _load(self):
        if self._model is not None:
            return
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        self.device = get_device()
        logger.info("Chargement PE-AV %s sur %s", self.model_id, self.device)
        self._processor = PeAudioVideoProcessor.from_pretrained(self.model_id)
        # transformers 5.21 + timm 1.0.30 : la config vision déclare `num_classes=1024` mais pas
        # `num_labels`, et TimmWrapper construit une tête à 2 classes → tailles incompatibles au
        # chargement. On aligne num_labels sur le checkpoint avant de charger les poids.
        from transformers import PeAudioVideoConfig
        cfg = PeAudioVideoConfig.from_pretrained(self.model_id)
        vis = getattr(getattr(cfg.video_config, "video_config", None), "vision_config", None)
        if vis is not None and getattr(vis, "num_labels", None) in (None, 2) and getattr(vis, "num_classes", None):
            vis.num_labels = int(vis.num_classes)
        self._model = PeAudioVideoModel.from_pretrained(self.model_id, config=cfg, dtype=torch.float32).to(self.device).eval()
        fe = getattr(self._processor, "feature_extractor", None) or getattr(self._processor, "audio_processor", None)
        self.sr = int(getattr(fe, "sampling_rate", 0) or 0)
        if not self.sr:
            raise RuntimeError("PE-AV : fréquence d'échantillonnage audio introuvable dans le processeur ; vérifier l'API")
        logger.info("PE-AV : audio attendu à %d Hz", self.sr)

    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=self.sr)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, self.sr)

    def _forward(self, **kw):
        inputs = self._processor(return_tensors="pt", padding=True, **kw)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            return self._model(**inputs)

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        out = self._forward(audio=[self._to_wave(it) for it in items], sampling_rate=self.sr)
        emb = getattr(out, "audio_embeds", None)
        if emb is None:
            raise RuntimeError(f"PE-AV : sortie sans audio_embeds ; clés : {list(out.keys())}")
        self.dim = emb.shape[-1]
        return emb.float().cpu().numpy()

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        self._load()
        out = self._forward(text=list(texts))
        key = "text_audio_embeds" if self.text_space == "audio" else "text_video_embeds"
        emb = getattr(out, key, None)
        if emb is None:
            raise RuntimeError(f"PE-AV : sortie sans {key} ; clés : {list(out.keys())}")
        self.dim = emb.shape[-1]
        return emb.float().cpu().numpy()

    def _embed_video(self, paths: Sequence[str]) -> np.ndarray:
        self._load()
        out = self._forward(videos=list(paths))
        emb = getattr(out, "video_embeds", None)
        if emb is None:
            raise RuntimeError(f"PE-AV : sortie sans video_embeds ; clés : {list(out.keys())}")
        return emb.float().cpu().numpy()
