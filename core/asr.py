"""Transcription de la voix chantée (paroles) avec faster-whisper (CTranslate2, CPU int8).

Utilisée quand l'entrée audio est une voix seule (fredonnement / chant) : le fragment de paroles
devient une requête texte (empreinte de paroles, Llama-retriever, fiches, web). Modèle réglable
par SOUNDTRACE_ASR_MODEL (défaut `small` : ≈ 0,5× temps réel sur M2 CPU). Cache par hash de l'audio.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from core.audio_io import load_audio, resample, to_mono
from core.cache import DiskCache, content_hash

logger = logging.getLogger(__name__)
_model = None


@dataclass
class Transcript:
    text: str
    language: str | None
    language_probability: float
    n_words: int


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        name = os.environ.get("SOUNDTRACE_ASR_MODEL", "small")
        logger.info("Chargement Whisper %s (CPU int8)", name)
        _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: bytes | tuple[np.ndarray, int], language: str | None = None) -> Transcript:
    if isinstance(audio, (bytes, bytearray)):
        key = content_hash(bytes(audio), kind="asr", model=os.environ.get("SOUNDTRACE_ASR_MODEL", "small"), v=1)
        x, _ = load_audio(bytes(audio), sr=16_000)
    else:
        x0, sr = audio
        x = resample(to_mono(np.asarray(x0)), sr, 16_000)
        key = content_hash(np.ascontiguousarray(x, dtype=np.float32).tobytes(), kind="asr", model=os.environ.get("SOUNDTRACE_ASR_MODEL", "small"), v=1)
    cache = DiskCache("asr")
    hit = cache.get(key)
    if hit is not None:
        return Transcript(**hit)
    segs, info = _get_model().transcribe(x.astype(np.float32), beam_size=1, vad_filter=False, language=language)
    text = " ".join(s.text.strip() for s in segs).strip()
    t = Transcript(text=text, language=info.language, language_probability=float(info.language_probability), n_words=len(text.split()))
    cache.put(key, t.__dict__)
    return t
