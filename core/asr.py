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
    avg_logprob: float = 0.0        # moyenne des log-probas des segments (Whisper) ; très bas = texte deviné
    no_speech_prob: float = 0.0     # proba « pas de parole » max sur les segments

    @property
    def has_words(self) -> bool:
        """Vrai si l'on a vraiment entendu des paroles (chantées), pas un « mmm » que Whisper habille d'une
        phrase inventée. Seuils posés à la main le 29/10 : un fredonnement sans mots donne une langue à ~0,35 de
        probabilité et des phrases de remplissage connues (« Thank you for watching! »)."""
        if self.n_words < 3 or is_hallucination(self.text):
            return False
        if self.language_probability < 0.5:
            return False
        return self.avg_logprob > -1.2 and self.no_speech_prob < 0.8


_HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "thank you", "merci d'avoir regardé", "sous-titres", "sous-titrage",
    "amara.org", "subtitles by", "sub by", "you", "bye", "music", "musique", "la la la", "mm", "hmm", "♪",
)


def is_hallucination(text: str) -> bool:
    """Phrases de remplissage typiques de Whisper sur de l'audio sans parole."""
    t = " ".join(text.lower().replace("!", " ").replace(".", " ").replace(",", " ").split())
    if not t:
        return True
    return any(t == h or t.startswith(h) and len(t) <= len(h) + 12 for h in _HALLUCINATIONS)


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
        key = content_hash(bytes(audio), kind="asr", model=os.environ.get("SOUNDTRACE_ASR_MODEL", "small"), v=2)
        x, _ = load_audio(bytes(audio), sr=16_000)
    else:
        x0, sr = audio
        x = resample(to_mono(np.asarray(x0)), sr, 16_000)
        key = content_hash(np.ascontiguousarray(x, dtype=np.float32).tobytes(), kind="asr", model=os.environ.get("SOUNDTRACE_ASR_MODEL", "small"), v=2)
    cache = DiskCache("asr")
    hit = cache.get(key)
    if hit is not None:
        return Transcript(**hit)
    segs, info = _get_model().transcribe(x.astype(np.float32), beam_size=1, vad_filter=False, language=language)
    segs = list(segs)
    text = " ".join(s.text.strip() for s in segs).strip()
    lp = [float(getattr(s, "avg_logprob", 0.0)) for s in segs]
    ns = [float(getattr(s, "no_speech_prob", 0.0)) for s in segs]
    t = Transcript(text=text, language=info.language, language_probability=float(info.language_probability), n_words=len(text.split()),
                   avg_logprob=float(np.mean(lp)) if lp else -9.0, no_speech_prob=max(ns) if ns else 1.0)
    cache.put(key, t.__dict__)
    return t
