"""Matching of a lyrics fragment (sung transcription) against the catalogue's lyrics
fingerprints (n-gram hashes, scripts/build_lyrics_hashes.py). No lyrics text on the index side.

Score of a track = number of n-grams of the fragment present in its fingerprint; returned
together with the share of covered n-grams. An approximate Whisper transcription often
lands exactly on a few 4-word n-grams: minimum threshold `min_hits`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.config import get_settings

logger = logging.getLogger(__name__)


def path() -> Path:
    return get_settings().index_dir / "lyrics_ngrams.parquet"


class LyricsHashIndex:
    def __init__(self, n: int = 4):
        import pyarrow.parquet as pq
        from scripts.build_lyrics_hashes import ngram_hashes, normalize_words  # reuses the same normalisation
        self._ngram_hashes, self._normalize = ngram_hashes, normalize_words
        self.n = n
        t = pq.read_table(path())
        self.inverted: dict[int, list[str]] = defaultdict(list)
        self.sizes: dict[str, int] = {}
        for tid, hs in zip(t.column("track_id").to_pylist(), t.column("hashes").to_pylist()):
            self.sizes[tid] = len(hs)
            for h in hs:
                self.inverted[h].append(tid)
        self._mtime = path().stat().st_mtime
        logger.info("lyrics fingerprints: %d tracks, %d distinct n-grams", len(self.sizes), len(self.inverted))

    @classmethod
    def exists(cls) -> bool:
        return path().exists()

    def search(self, text: str, k: int = 20, min_hits: int = 2) -> list[tuple[str, float]]:
        words = self._normalize(text)
        qh = self._ngram_hashes(words, self.n)
        if not len(qh):
            return []
        hits: dict[str, int] = defaultdict(int)
        for h in qh.tolist():
            for tid in self.inverted.get(h, ()):
                hits[tid] += 1
        scored = [(tid, c + 0.001 * c / max(len(qh), 1)) for tid, c in hits.items() if c >= min_hits]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
