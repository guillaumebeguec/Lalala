"""Empreinte robuste au micro — audfprint (Dan Ellis, MIT), recommandation n° 8.

Empreinte par paires de pics spectraux (landmarks, type Shazam) : contrairement à Chromaprint
(conçu pour comparer des fichiers), elle résiste à la capture au micro, au bruit et aux
extraits partiels. Code `dpwe/audfprint` cloné dans data/raw/audfprint/repo (`AUDFPRINT_DIR`).

API lue le 29/10/2026 : `Analyzer(density).wavfile2hashes(fichier)` → paires (temps, hash) ;
`HashTable.store(nom, hashes)` / `get_hits` ; `Matcher.match_hashes(ht, hashes)` → liste de
(id, hashes alignés, décalage, hashes bruts, rang, tmin, tmax). Score = hashes alignés ; au-delà
de `MIN_ALIGNED` (défaut 12) la correspondance est quasi certaine. Audio à 11 025 Hz ; les
octets mp3/wav/webm passent par un fichier temporaire (audio_read/ffmpeg).

Base persistée dans data/index/audfprint.pklz (2^20 hachages, profondeur 100 : ~100 k morceaux).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.audio_io import load_audio, to_wav_bytes
from core.config import get_settings

logger = logging.getLogger(__name__)

MIN_ALIGNED = 10      # hashes alignés au-delà desquels on tient la correspondance pour sûre
DENSITY = 20.0
QUERY_SHIFTS = 4      # décalages sous-trame côté requête : indispensable sur extraits coupés / capturés (0/10 → 6/10 mesuré)
THRESHCOUNT = 3


def audfprint_dir() -> Path:
    return Path(os.environ.get("AUDFPRINT_DIR", get_settings().data_dir / "raw" / "audfprint" / "repo"))


def _import():
    d = str(audfprint_dir())
    if not (audfprint_dir() / "audfprint_analyze.py").exists():
        raise RuntimeError(f"audfprint absent : {d} (git clone https://github.com/dpwe/audfprint)")
    if d not in sys.path:
        sys.path.insert(0, d)
    import audfprint_analyze, hash_table, audfprint_match  # type: ignore  # noqa: E401
    return audfprint_analyze, hash_table, audfprint_match


class AudfprintIndex:
    def __init__(self, density: float = DENSITY, hashbits: int = 20, depth: int = 100):
        an, ht, mt = _import()
        self._an = an.Analyzer(density=density)
        self._an.fail_on_error = False
        self.ht = ht.HashTable(hashbits=hashbits, depth=depth, maxtime=16384)
        self._matcher = mt.Matcher()
        self._matcher.max_returns = 10
        self._matcher.search_depth = 100
        self._matcher.threshcount = THRESHCOUNT
        self._matcher.verbose = False

    @classmethod
    def path(cls) -> Path:
        return get_settings().index_dir / "audfprint.pklz"

    # ---- empreintes ----
    def hashes(self, audio: bytes | tuple[np.ndarray, int] | str | Path) -> np.ndarray:
        """Paires (temps, hash) d'un audio (octets, (signal, sr) ou chemin)."""
        if isinstance(audio, (str, Path)):
            return self._an.wavfile2hashes(str(audio))
        if isinstance(audio, tuple):
            x, sr = audio
            data, suffix = to_wav_bytes(np.asarray(x, dtype=np.float32), sr), ".wav"
        else:
            x, sr = load_audio(bytes(audio), sr=11_025)   # décodage robuste (mp3, webm, m4a) puis wav
            data, suffix = to_wav_bytes(x, sr), ".wav"
        fd, p = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return self._an.wavfile2hashes(p)
        finally:
            os.unlink(p)

    def add(self, track_id: str, audio) -> int:
        h = self.hashes(audio)
        if h is None or len(h) == 0:
            return 0
        self.ht.store(track_id, h)
        return int(len(h))

    def __contains__(self, track_id: str) -> bool:
        try:
            return self.ht.name_to_id(track_id, add_if_missing=False) is not None and track_id in self.ht.names
        except Exception:  # noqa: BLE001
            return track_id in getattr(self.ht, "names", [])

    def search(self, audio, k: int = 5) -> list[tuple[str, float, int]]:
        """[(track_id, score = hashes alignés, décalage en trames)] triés. Vide si rien n'atteint threshcount.
        La requête est hachée avec `QUERY_SHIFTS` décalages sous-trame (la référence sans)."""
        self._an.shifts = QUERY_SHIFTS
        try:
            h = self.hashes(audio)
        finally:
            self._an.shifts = 1
        if h is None or len(h) == 0:
            return []
        self._matcher.max_returns = k
        res = self._matcher.match_hashes(self.ht, h)
        out = []
        for row in res:
            idx, aligned, offset = int(row[0]), int(row[1]), int(row[2])
            if aligned <= 0:
                continue
            out.append((self.ht.names[idx], float(aligned), offset))
        return out

    # ---- persistance ----
    def save(self) -> Path:
        p = self.path(); p.parent.mkdir(parents=True, exist_ok=True)
        self.ht.save(str(p))
        return p

    @classmethod
    def load(cls) -> "AudfprintIndex":
        idx = cls()
        idx.ht.load(str(cls.path()))
        return idx

    def __len__(self) -> int:
        return len([n for n in getattr(self.ht, "names", []) if n])
