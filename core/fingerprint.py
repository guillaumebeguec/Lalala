"""Empreinte audio Chromaprint (baseline type Shazam) — PLAN.md phase 3.6.

`fpcalc` (Chromaprint, Homebrew) calcule une empreinte = suite d'entiers 32 bits
(une valeur ≈ 1/8 s). On compare une requête à chaque extrait du catalogue par
le **meilleur alignement** (décalage) qui minimise la distance de Hamming
moyenne sur la zone de recouvrement. Score ∈ [0, 1] : 1 = identique. Au-delà
d'environ 0,85 sur ≥ 4 s de recouvrement, c'est le même enregistrement.

Les empreintes du catalogue sont stockées dans data/index/fingerprints.npz
(objets : track_id → np.uint32[]). Rien d'audio n'est conservé.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.audio_io import to_wav_bytes
from core.config import get_settings

logger = logging.getLogger(__name__)

FP_RATE = 8.0  # valeurs par seconde (approx. Chromaprint par défaut)


def fpcalc_raw(audio: bytes | tuple[np.ndarray, int] | str | Path, max_length_s: int = 60) -> np.ndarray:
    """Empreinte brute via `fpcalc -raw`. Accepte octets (mp3/wav), (signal, sr) ou chemin."""
    import acoustid
    if isinstance(audio, (str, Path)):
        _, fp = acoustid.fingerprint_file(str(audio), maxlength=max_length_s)
        return _decode(fp)
    if isinstance(audio, tuple):
        x, sr = audio
        data, suffix = to_wav_bytes(x, sr), ".wav"
    else:
        data = bytes(audio)
        suffix = ".mp3" if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") else ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        out = subprocess.run(["fpcalc", "-raw", "-length", str(max_length_s), path], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    for line in out.splitlines():
        if line.startswith("FINGERPRINT="):
            return np.array([int(v) for v in line[len("FINGERPRINT="):].split(",") if v], dtype=np.uint32)
    raise RuntimeError("fpcalc : pas d'empreinte dans la sortie")


def _decode(fp) -> np.ndarray:
    """Empreinte compressée (chaîne base64 de fpcalc) → brute, via pyacoustid/chromaprint."""
    import chromaprint
    raw, _algo = chromaprint.decode_fingerprint(fp if isinstance(fp, bytes) else fp.encode())
    return np.array(raw, dtype=np.uint32)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> float:
    x = np.bitwise_xor(a, b).view(np.uint8)
    return float(_POPCOUNT[x].sum()) / (32.0 * len(a))


def match_score(query: np.ndarray, ref: np.ndarray, min_overlap: int = 32) -> tuple[float, int]:
    """Meilleur (1 − distance de Hamming moyenne) sur tous les décalages avec ≥ min_overlap valeurs."""
    q, r = query.astype(np.uint32), ref.astype(np.uint32)
    best, best_off = 0.0, 0
    for off in range(-(len(q) - min_overlap), len(r) - min_overlap + 1):
        if off >= 0:
            a, b = q[: len(r) - off], r[off : off + len(q)]
        else:
            a, b = q[-off : -off + len(r)], r[: len(q) + off]
        n = min(len(a), len(b))
        if n < min_overlap:
            continue
        s = 1.0 - _hamming(a[:n], b[:n])
        if s > best:
            best, best_off = s, off
    return best, best_off


@dataclass
class FingerprintIndex:
    fps: dict[str, np.ndarray]

    @classmethod
    def path(cls) -> Path:
        return get_settings().index_dir / "fingerprints.npz"

    def save(self) -> Path:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{k: v for k, v in self.fps.items()})
        return p

    @classmethod
    def load(cls) -> "FingerprintIndex":
        with np.load(cls.path()) as z:
            return cls({k: z[k] for k in z.files})

    def search(self, query_fp: np.ndarray, k: int = 5, candidates: list[str] | None = None) -> list[tuple[str, float]]:
        ids = candidates if candidates is not None else list(self.fps)
        scored = [(tid, match_score(query_fp, self.fps[tid])[0]) for tid in ids if tid in self.fps]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
