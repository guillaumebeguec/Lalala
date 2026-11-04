"""hum2song — query by humming (PLAN_COMPLEMENTS.md C4, replaces phase 9.5).

Model `luigir5/hum2song` (checkpoint `e2b`, CC BY 4.0 weights, ~5.2M parameters): melodic
contour encoder, a single tower for queries and tracks, normalised 256 d embeddings.
Code `luiginotmario/hum2song` cloned in `data/raw/hum2song/repo` (`HUM2SONG_DIR`); the
pitch tracker is RMVPE (vendored RVC build `docs/paper/scripts/rmvpe_rvc.py`, weights
`rmvpe.pt` from `lj1995/VoiceConversionWebUI`), at 16 kHz, 10 ms per frame.

Pipeline read from the HF card and `training/src/hum2song/contour/` (28/10/2026):
16 kHz mono audio → `trackers.rmvpe_track` (2, frames) [F0 Hz, salience] →
`features.rmvpe_contour` (key-normalised contour, 20 ms) → queries: `evaluate.query_contour`
(<= 20 s, silences trimmed); tracks: full contour of the 30 s preview → `evaluate.embed_contours`.

Uses neither timbre nor lyrics; on the catalogue side, the contour is extracted from the full mix
(without Demucs separation at first, see C4.2: to be compared).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from core.audio_io import load_audio, resample, to_mono
from core.config import get_settings
from core.embedders.base import AudioItem, Embedder, l2_normalize

logger = logging.getLogger(__name__)

SR = 16_000


def hum2song_dir() -> Path:
    return Path(os.environ.get("HUM2SONG_DIR", get_settings().data_dir / "raw" / "hum2song" / "repo"))


class Hum2SongEmbedder(Embedder):
    name = "hum2song"
    dim = 256
    supports_video = False
    languages = "none"

    def __init__(self, role: str = "song", checkpoint: str = "e2b", device: str | None = None,
                 separate_vocals: bool | None = None, windows: bool | None = None, win_s: float = 12.0, hop_s: float = 3.0):
        super().__init__()
        self.role = role          # "song": contour of the preview; "query": trimmed to 20 s (humming)
        self.checkpoint = checkpoint
        self._device_name = device or "cpu"   # RMVPE (RVC) and the encoder are lightweight: CPU by default, stable
        env = os.environ
        # C4.2: on the track side, vocals separated by Demucs (htdemucs) and/or sliding windows (score = max)
        self.separate_vocals = (env.get("HUM2SONG_DEMUCS", "0") == "1") if separate_vocals is None else separate_vocals
        self.windows = (env.get("HUM2SONG_WINDOWS", "0") == "1") if windows is None else windows
        self.win_s, self.hop_s = win_s, hop_s
        self._model = None
        self._rmvpe = None
        self._demucs = None

    def version(self) -> str:
        return f"1-{self.checkpoint}-{self.role}-{'voc' if self.separate_vocals and self.role == 'song' else 'mix'}-{'win' if self.windows and self.role == 'song' else 'full'}"

    def _load(self):
        if self._model is not None:
            return
        root = hum2song_dir()
        src = root / "training" / "src"
        if not (src / "hum2song").exists():
            raise RuntimeError(f"hum2song missing: {src} (see PLAN_COMPLEMENTS.md C4)")
        for p in (str(src), str(root / "docs" / "paper" / "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from huggingface_hub import hf_hub_download
        from hum2song.contour.train import load_checkpoint  # type: ignore
        from hum2song.contour import trackers  # type: ignore
        self.device = torch.device(self._device_name)
        ckpt = Path(hf_hub_download("luigir5/hum2song", f"{self.checkpoint}/best.pt"))
        self._model, self.config, step = load_checkpoint(ckpt, self.device)
        weights = Path(hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt"))
        self._rmvpe = trackers.load_rmvpe(weights, self.device)
        self._trackers = trackers
        logger.info("hum2song %s loaded (step %d) on %s", self.checkpoint, step, self.device)

    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=SR)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, SR)

    def _vocals(self, item: AudioItem) -> np.ndarray:
        """Vocals separated by Demucs (htdemucs, 44.1 kHz stereo) then brought back to 16 kHz mono."""
        if self._demucs is None:
            from demucs.pretrained import get_model
            self._demucs = get_model("htdemucs").eval()
        from demucs.apply import apply_model
        if isinstance(item, (bytes, bytearray)):
            x, sr = load_audio(bytes(item), sr=44_100, mono=False)
        else:
            x, sr = item
            x = resample(to_mono(np.asarray(x)), sr, 44_100)
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        wav = torch.from_numpy(x.T.astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            src = apply_model(self._demucs, wav, device=torch.device("cpu"), progress=False)[0]
        v = src[self._demucs.sources.index("vocals")].mean(0).numpy()
        return resample(v.astype(np.float32), 44_100, SR)

    def pitch_track(self, item: AudioItem) -> np.ndarray:
        """RMVPE pitch track of the mix: (2, 10 ms frames) = [F0 Hz, salience 0–1]."""
        self._load()
        return self._trackers.rmvpe_track(self._rmvpe, self._to_wave(item).astype(np.float32))

    def contour(self, item: AudioItem) -> np.ndarray:
        """Normalised melodic contour (20 ms frames) of an audio, via RMVPE."""
        self._load()
        from hum2song.contour.features import rmvpe_contour  # type: ignore
        from hum2song.contour.evaluate import query_contour  # type: ignore
        x = (self._vocals(item) if (self.role == "song" and self.separate_vocals) else self._to_wave(item)).astype(np.float32)
        track = self._trackers.rmvpe_track(self._rmvpe, x)
        c = rmvpe_contour(track)
        return query_contour(c) if self.role == "query" else c

    def _windows(self, c: np.ndarray) -> list[np.ndarray]:
        from hum2song.contour.evaluate import query_contour  # type: ignore
        win, hop = int(self.win_s / 0.02), int(self.hop_s / 0.02)
        if len(c) <= win:
            return [query_contour(c)]
        return [query_contour(c[s : s + win]) for s in range(0, len(c) - win + 1, hop)]

    def embed_windows(self, item: AudioItem) -> np.ndarray:
        """Embeddings (n_windows, 256) of a track; query→track score = max of similarities."""
        self._load()
        from hum2song.contour.evaluate import embed_contours  # type: ignore
        c = self.contour(item)
        if not len(c):
            return np.zeros((0, self.dim), np.float32)
        ws = [w for w in self._windows(c) if len(w)]
        return l2_normalize(np.asarray(embed_contours(self._model, ws, self.device), dtype=np.float32)) if ws else np.zeros((0, self.dim), np.float32)

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        from hum2song.contour.evaluate import embed_contours  # type: ignore
        out = np.zeros((len(items), self.dim), np.float32)
        if self.role == "song" and self.windows:
            # a single vector per track in the FAISS index: normalised mean of the windows
            # (the windows themselves are used for the max score via `embed_windows`)
            for i, it in enumerate(items):
                W = self.embed_windows(it)
                if len(W):
                    out[i] = l2_normalize(W.mean(axis=0))[0]
            return out
        contours = [self.contour(it) for it in items]
        keep = [i for i, c in enumerate(contours) if len(c)]
        if keep:
            E = np.asarray(embed_contours(self._model, [contours[i] for i in keep], self.device), dtype=np.float32)
            for j, i in enumerate(keep):
                out[i] = E[j]
        return out

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        raise NotImplementedError("hum2song does not encode text")
