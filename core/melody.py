"""Direct melodic matching (hummed query <-> catalogue contours), without a learned model.

Finding from 29/10: hum2song (contour embeddings, trained on full songs + separated
vocals) retrieves < 3% of CHAD hums and even misses *synthetic* hums rendered from
the melody of the preview itself. Here we go back to the classic query-by-humming
method: align the pitch contour of the query onto the contours of the previews, with
tolerance to **transposition** (the voice does not sing in the key of the record), to
**tempo** (factors 0.8–1.25) and to **gaps** (unvoiced frames ignored).

- Contours: RMVPE (same tracker as hum2song), 20 ms frames, pitch in MIDI semitones,
  NaN when the frame is unvoiced. Catalogue: `ContourStore` (`data/index/contours_<subset>.npz`,
  built by `scripts/build_contours.py`).
- Score of a window = 1 − error / 3, where the error is the truncated mean (at 3 semitones) of
  |contour − query − transposition| over the frames voiced on both sides; the transposition is
  the mean of the differences, refined once by discarding frames more than 2 semitones away (octave
  errors). Track score = max over positions (100 ms hop) and tempos.

Cost: ≈ 4–8 s per 10 s query over 2,000 previews (numpy, batched); contours cached
by audio content hash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import get_settings

logger = logging.getLogger(__name__)

FRAME_S = 0.02            # contour hop (RMVPE outputs 10 ms, averaged by pairs)
MIN_CONF = 0.3            # minimum RMVPE salience for a voiced frame
F0_MIN, F0_MAX = 60.0, 1100.0
TEMPOS = (0.8, 0.9, 1.0, 1.12, 1.25)
HOP = 5                   # sliding hop over the catalogue, in decimated frames (5 x 40 ms = 200 ms)
DECIM = 2                 # 40 ms frames at matching time
MAX_DEV = 3.0             # deviation (semitones) beyond which a frame counts as "wrong"
MIN_OVERLAP = 0.5         # minimum share of the query's voiced frames covered by the window
MIN_QUERY_FRAMES = 20     # at least 0.8 s voiced (decimated frames)


def contour_from_track(track: np.ndarray, min_conf: float = MIN_CONF) -> np.ndarray:
    """(2, 10 ms frames) [F0 Hz, salience] → MIDI semitones at 20 ms, NaN if unvoiced."""
    f0, conf = np.asarray(track[0], dtype=np.float64), np.asarray(track[1], dtype=np.float64)
    voiced = (conf > min_conf) & (f0 > F0_MIN) & (f0 < F0_MAX)
    semi = np.where(voiced, 12.0 * np.log2(np.maximum(f0, 1.0) / 440.0) + 69.0, np.nan)
    n = (len(semi) // 2) * 2
    pairs = semi[:n].reshape(-1, 2)
    with np.errstate(all="ignore"):
        out = np.nanmean(pairs, axis=1)   # one voiced frame out of two is enough
    return out.astype(np.float32)


def _resample_tempo(q: np.ndarray, tempo: float) -> np.ndarray:
    """Stretch/compress the query: tempo 1.25 = the person sang faster than the record."""
    m = max(1, int(round(len(q) * tempo)))
    idx = np.clip(np.round(np.linspace(0, len(q) - 1, m)).astype(int), 0, len(q) - 1)
    return q[idx]


@dataclass
class ContourStore:
    ids: list[str]
    data: np.ndarray          # (n, T) float32, NaN = unvoiced or beyond the length

    @staticmethod
    def path(subset: str, provider: str | None = None) -> Path:
        suffix = f"_{provider}" if provider else ""
        return get_settings().index_dir / f"contours_{subset}{suffix}.npz"

    @classmethod
    def paths(cls, subset: str) -> list[Path]:
        """Main store (Deezer previews) + extra windows (`contours_<subset>_itunes.npz`...)."""
        base = cls.path(subset)
        extra = sorted(base.parent.glob(f"contours_{subset}_*.npz"))
        return [p for p in [base] + extra if p.exists()]

    @classmethod
    def exists(cls, subset: str) -> bool:
        return cls.path(subset).exists()

    @classmethod
    def mtime(cls, subset: str) -> float:
        return max((p.stat().st_mtime for p in cls.paths(subset)), default=0.0)

    @classmethod
    def load(cls, subset: str, merge: bool = True) -> "ContourStore":
        """Load the store; with `merge`, concatenate the extra windows (the same id may
        appear several times: `match` keeps the best alignment per track)."""
        paths = cls.paths(subset) if merge else [cls.path(subset)]
        ids: list[str] = []
        blocks: list[np.ndarray] = []
        for p in paths:
            z = np.load(p, allow_pickle=False)
            ids += [str(i) for i in z["ids"]]
            blocks.append(z["data"].astype(np.float32))
        T = max(b.shape[1] for b in blocks)
        data = np.full((len(ids), T), np.nan, dtype=np.float32)
        r = 0
        for b in blocks:
            data[r : r + len(b), : b.shape[1]] = b
            r += len(b)
        return cls(ids=ids, data=data)

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
    """D (b, T), q (m,) → best score (b,) and position (b,) over the sliding windows."""
    b, T = D.shape
    m = len(q)
    qv = ~np.isnan(q)
    nq = int(qv.sum())
    if T < m:
        pad = np.full((b, m - T), np.nan, dtype=np.float32)
        D = np.concatenate([D, pad], axis=1)
        T = m
    W = np.lib.stride_tricks.sliding_window_view(D, m, axis=1)[:, ::hop, :]   # (b, P, m) view without copy
    with np.errstate(all="ignore"):
        diff = W - q                                                # NaN if either side is unvoiced
        both = ~np.isnan(diff)
        overlap = both.sum(axis=2)                                  # (b, P)
        d0 = np.where(both, diff, 0.0)
        shift = d0.sum(axis=2) / np.maximum(overlap, 1)             # transposition: mean of the differences
        keep = both & (np.abs(diff - shift[..., None]) < 2.0)       # refinement: discard octave errors
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
    """Ranking of the catalogue for a query contour: [(track_id, score 0–1, position s)].
    `decim` subsamples query and catalogue (2 → 40 ms frames, ≈ 4x faster)."""
    q = np.asarray(query, dtype=np.float32)[::decim]
    data = store.data[:, ::decim]
    voiced = ~np.isnan(q)
    if voiced.sum() < MIN_QUERY_FRAMES:
        logger.info("melody: query has too few voiced frames (%d frames)", int(voiced.sum()))
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
    # a track present under several windows (Deezer, iTunes): keep its best alignment
    seen: dict[str, int] = {}
    for i in np.argsort(-best):
        if best[i] <= 0:
            break
        tid = store.ids[i]
        if tid not in seen:
            seen[tid] = i
            if len(seen) >= top:
                break
    return [(tid, float(best[i]), float(pos[i] * FRAME_S * decim)) for tid, i in seen.items()]
