"""Appariement mélodique direct (requête fredonnée ↔ contours du catalogue), sans modèle appris.

Constat du 29/10 : hum2song (embeddings de contours, entraîné sur chansons complètes + voix
séparée) retrouve < 3 % des fredonnements CHAD et rate même des fredonnements *synthétiques*
rendus depuis la mélodie de l'extrait lui-même. Ici on revient à la méthode classique de la
recherche par fredonnement (query-by-humming) : aligner le contour de hauteur de la requête
sur les contours des extraits, avec tolérance à la **transposition** (la voix ne chante pas
dans la tonalité du disque), au **tempo** (facteurs 0,8–1,25) et aux **trous** (trames non
voisées ignorées).

- Contours : RMVPE (même traceur que hum2song), trames de 20 ms, hauteur en demi-tons MIDI,
  NaN quand la trame n'est pas voisée. Catalogue : `ContourStore` (`data/index/contours_<subset>.npz`,
  construit par `scripts/build_contours.py`).
- Score d'une fenêtre = 1 − erreur / 3, où l'erreur est la moyenne tronquée (à 3 demi-tons) de
  |contour − requête − transposition| sur les trames voisées des deux côtés ; la transposition est
  la moyenne des écarts, raffinée une fois en écartant les trames à plus de 2 demi-tons (erreurs
  d'octave). Score du morceau = max sur les positions (pas de 100 ms) et les tempos.

Coût : ≈ 4–8 s par requête de 10 s sur 2 000 extraits (numpy, par lots) ; cache des contours
par hash du contenu audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import get_settings

logger = logging.getLogger(__name__)

FRAME_S = 0.02            # pas des contours (RMVPE sort du 10 ms, on moyenne par deux)
MIN_CONF = 0.3            # saillance RMVPE minimale pour une trame voisée
F0_MIN, F0_MAX = 60.0, 1100.0
TEMPOS = (0.8, 0.9, 1.0, 1.12, 1.25)
HOP = 5                   # pas de glissement sur le catalogue, en trames décimées (5 × 40 ms = 200 ms)
DECIM = 2                 # trames de 40 ms à l'appariement
MAX_DEV = 3.0             # écart (demi-tons) au-delà duquel une trame compte comme « fausse »
MIN_OVERLAP = 0.5         # part minimale des trames voisées de la requête couvertes par la fenêtre
MIN_QUERY_FRAMES = 20     # 0,8 s voisée au moins (trames décimées)


def contour_from_track(track: np.ndarray, min_conf: float = MIN_CONF) -> np.ndarray:
    """(2, trames 10 ms) [F0 Hz, saillance] → demi-tons MIDI à 20 ms, NaN si non voisé."""
    f0, conf = np.asarray(track[0], dtype=np.float64), np.asarray(track[1], dtype=np.float64)
    voiced = (conf > min_conf) & (f0 > F0_MIN) & (f0 < F0_MAX)
    semi = np.where(voiced, 12.0 * np.log2(np.maximum(f0, 1.0) / 440.0) + 69.0, np.nan)
    n = (len(semi) // 2) * 2
    pairs = semi[:n].reshape(-1, 2)
    with np.errstate(all="ignore"):
        out = np.nanmean(pairs, axis=1)   # une trame sur deux voisée suffit
    return out.astype(np.float32)


def _resample_tempo(q: np.ndarray, tempo: float) -> np.ndarray:
    """Étire/compresse la requête : tempo 1,25 = la personne a chanté plus vite que le disque."""
    m = max(1, int(round(len(q) * tempo)))
    idx = np.clip(np.round(np.linspace(0, len(q) - 1, m)).astype(int), 0, len(q) - 1)
    return q[idx]


@dataclass
class ContourStore:
    ids: list[str]
    data: np.ndarray          # (n, T) float32, NaN = non voisé ou au-delà de la longueur

    @staticmethod
    def path(subset: str) -> Path:
        return get_settings().index_dir / f"contours_{subset}.npz"

    @classmethod
    def exists(cls, subset: str) -> bool:
        return cls.path(subset).exists()

    @classmethod
    def load(cls, subset: str) -> "ContourStore":
        z = np.load(cls.path(subset), allow_pickle=False)
        return cls(ids=[str(i) for i in z["ids"]], data=z["data"].astype(np.float32))

    @classmethod
    def build(cls, contours: dict[str, np.ndarray]) -> "ContourStore":
        ids = list(contours)
        T = max((len(c) for c in contours.values()), default=0)
        data = np.full((len(ids), T), np.nan, dtype=np.float32)
        for i, tid in enumerate(ids):
            c = contours[tid]
            data[i, : len(c)] = c
        return cls(ids=ids, data=data)

    def save(self, subset: str) -> Path:
        p = self.path(subset)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, ids=np.asarray(self.ids), data=self.data)
        return p

    def __len__(self) -> int:
        return len(self.ids)


def _score_batch(D: np.ndarray, q: np.ndarray, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """D (b, T), q (m,) → meilleur score (b,) et position (b,) sur les fenêtres glissantes."""
    b, T = D.shape
    m = len(q)
    qv = ~np.isnan(q)
    nq = int(qv.sum())
    if T < m:
        pad = np.full((b, m - T), np.nan, dtype=np.float32)
        D = np.concatenate([D, pad], axis=1)
        T = m
    W = np.lib.stride_tricks.sliding_window_view(D, m, axis=1)[:, ::hop, :]   # (b, P, m) vue sans copie
    with np.errstate(all="ignore"):
        diff = W - q                                                # NaN si l'un des deux n'est pas voisé
        both = ~np.isnan(diff)
        overlap = both.sum(axis=2)                                  # (b, P)
        d0 = np.where(both, diff, 0.0)
        shift = d0.sum(axis=2) / np.maximum(overlap, 1)             # transposition : moyenne des écarts
        keep = both & (np.abs(diff - shift[..., None]) < 2.0)       # raffinement : on écarte les erreurs d'octave
        n_keep = keep.sum(axis=2)
        shift = np.where(n_keep > 0, np.where(keep, diff, 0.0).sum(axis=2) / np.maximum(n_keep, 1), shift)
        dev = np.minimum(np.abs(diff - shift[..., None]), MAX_DEV)
        err = np.where(both, dev, 0.0).sum(axis=2) / np.maximum(overlap, 1)
        score = 1.0 - err / MAX_DEV
        score = np.where(overlap >= MIN_OVERLAP * nq, score, -1.0)
    best = score.argmax(axis=1)
    return score[np.arange(b), best], best * hop


def match(query: np.ndarray, store: ContourStore, top: int = 20, tempos: tuple[float, ...] = TEMPOS,
          hop: int = HOP, batch: int = 128, decim: int = DECIM) -> list[tuple[str, float, float]]:
    """Classement du catalogue pour un contour de requête : [(track_id, score 0–1, position s)].
    `decim` sous-échantillonne requête et catalogue (2 → trames de 40 ms, ≈ 4× plus rapide)."""
    q = np.asarray(query, dtype=np.float32)[::decim]
    data = store.data[:, ::decim]
    voiced = ~np.isnan(q)
    if voiced.sum() < MIN_QUERY_FRAMES:
        logger.info("mélodie : requête trop peu voisée (%d trames)", int(voiced.sum()))
        return []
    first, last = np.argmax(voiced), len(q) - np.argmax(voiced[::-1])
    q = q[first:last]
    n = len(store.ids)
    best = np.full(n, -1.0, dtype=np.float32)
    pos = np.zeros(n, dtype=np.int64)
    for tempo in tempos:
        qt = _resample_tempo(q, tempo)
        for s in range(0, n, batch):
            sc, p = _score_batch(data[s : s + batch], qt, hop)
            better = sc > best[s : s + batch]
            best[s : s + batch] = np.where(better, sc, best[s : s + batch])
            pos[s : s + batch] = np.where(better, p, pos[s : s + batch])
    order = np.argsort(-best)[:top]
    return [(store.ids[i], float(best[i]), float(pos[i] * FRAME_S * decim)) for i in order if best[i] > 0]
