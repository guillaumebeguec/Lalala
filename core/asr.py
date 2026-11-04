"""Transcription of the sung voice (lyrics) with faster-whisper (CTranslate2, CPU int8).

Used when the audio input is a lone voice (humming / singing): the lyrics fragment
becomes a text query (lyrics fingerprint, Llama retriever, cards, web). Model configurable
via LALALA_ASR_MODEL (default `small`: ≈ 0.5x real time on M2 CPU). Cached by audio hash.
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
    avg_logprob: float = 0.0        # mean of the segments' log-probs (Whisper); very low = guessed text
    no_speech_prob: float = 0.0     # max "no speech" probability over the segments

    @property
    def has_words(self) -> bool:
        """True if we really heard (sung) lyrics, not an "mmm" that Whisper dresses up as an
        invented sentence. Thresholds set by hand on 29/10: a wordless hum yields a language at ~0.35
        probability and well-known filler sentences ("Thank you for watching!")."""
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
    """Typical Whisper filler sentences on audio without speech."""
    t = " ".join(text.lower().replace("!", " ").replace(".", " ").replace(",", " ").split())
    if not t:
        return True
    return any(t == h or t.startswith(h) and len(t) <= len(h) + 12 for h in _HALLUCINATIONS)


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        name = os.environ.get("LALALA_ASR_MODEL", "small")
        logger.info("Loading Whisper %s (CPU int8)", name)
        _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: bytes | tuple[np.ndarray, int], language: str | None = None) -> Transcript:
    if isinstance(audio, (bytes, bytearray)):
        key = content_hash(bytes(audio), kind="asr", model=os.environ.get("LALALA_ASR_MODEL", "small"), v=2)
        x, _ = load_audio(bytes(audio), sr=16_000)
    else:
        x0, sr = audio
        x = resample(to_mono(np.asarray(x0)), sr, 16_000)
        key = content_hash(np.ascontiguousarray(x, dtype=np.float32).tobytes(), kind="asr", model=os.environ.get("LALALA_ASR_MODEL", "small"), v=2)
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
