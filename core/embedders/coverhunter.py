"""CoverHunter — identification de reprises (PLAN_COMPLEMENTS.md C2), audio → audio uniquement.

Code : `alanngnet/CoverHunterMPS` (support MPS), cloné dans `data/raw/coverhunter/CoverHunterMPS`
(`COVERHUNTER_DIR`) ; poids : checkpoint CoverHunter original (Liu Feng, entraîné sur SHS100K,
~65 Mo, Google Drive) dans `data/raw/coverhunter/pretrained/pretrain_model`. **Ni le code ni le
checkpoint n'indiquent de licence** : usage expérimental local, à vérifier avant toute diffusion.

Pipeline reproduit depuis `tools/identify.py` et `tools/make_embeds.py` (lus le 28/10/2026) :
audio mono 16 kHz, normalisation crête, CQT nnAudio (fmin 32 Hz, 96 bins, 12/octave, hop 640 =
25 trames/s), dB relatifs au max, fenêtre de `chunk_frame[0] × mean_size` = 3 375 trames (135 s)
rembourrée à −100 dB, moyenne par blocs de 3 trames → (1125, 96) → `Model.inference` → 128 d.

Deux stratégies (C2.3) : `window="full"` (un embedding par extrait) ou `window="sliding"`
(fenêtres de 10 s, pas de 5 s ; l'embedding renvoyé est la moyenne normalisée, et
`embed_windows` donne toutes les fenêtres pour un score = max des similarités).
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
from core.device import get_device
from core.embedders.base import AudioItem, Embedder, l2_normalize

logger = logging.getLogger(__name__)

SR = 16_000
FPS = 25  # trames CQT par seconde (hop 640 à 16 kHz)


def coverhunter_dir() -> Path:
    return Path(os.environ.get("COVERHUNTER_DIR", get_settings().data_dir / "raw" / "coverhunter"))


class CoverHunterEmbedder(Embedder):
    name = "coverhunter"
    dim = 128
    supports_video = False
    languages = "none"

    def __init__(self, window: str = "full", win_s: float = 10.0, hop_s: float = 5.0):
        super().__init__()
        self.window = window
        self.win_s, self.hop_s = win_s, hop_s
        self._model = None
        self._cqt = None
        self.hp: dict = {}

    def version(self) -> str:
        return f"1-{self.window}-{self.win_s}-{self.hop_s}"

    # ---- chargement ----
    def _load(self):
        if self._model is not None:
            return
        root = coverhunter_dir()
        code, ckpt = root / "CoverHunterMPS", root / "pretrained" / "pretrain_model"
        if not (code / "src" / "model.py").exists() or not (ckpt / "config" / "hparams.yaml").exists():
            raise RuntimeError(f"CoverHunter absent : attendu {code} et {ckpt} (voir PLAN_COMPLEMENTS.md C2)")
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        import yaml
        from src.model import Model  # type: ignore
        from nnAudio.features.cqt import CQT
        self.device = get_device()
        hp = yaml.safe_load((ckpt / "config" / "hparams.yaml").read_text())
        if "foc" not in hp and "ce" in hp:   # CoverHunterMPS a renommé `ce` en `foc`
            hp["foc"] = hp.pop("ce")
        hp["device"] = str(self.device)
        self.hp = hp
        model = Model(hp).to(self.device)
        model.load_model_parameters(str(ckpt / "pt_model"), device=str(self.device))
        model.eval()
        self._model = model
        self._cqt = CQT(SR, hop_length=640, n_bins=96, fmin=32, bins_per_octave=12, verbose=False).to(self.device)
        cf = hp["chunk_frame"]
        self.chunk_frames = (cf[0] if isinstance(cf, list) else cf) * hp["mean_size"]
        self.mean_size = hp["mean_size"]
        self.dim = int(hp["embed_dim"])
        logger.info("CoverHunter chargé sur %s (%d paramètres)", self.device, model.model_size())

    # ---- caractéristiques ----
    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=SR)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, SR)

    @torch.inference_mode()
    def _cqt_db(self, x: np.ndarray) -> np.ndarray:
        """(trames, 96) en dB relatifs, comme `_make_feat` de tools/identify.py."""
        sig = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)
        sig = sig / torch.max(torch.tensor(0.001, device=self.device), sig.abs().max()) * 0.999
        spec = self._cqt(sig) + 1e-9
        spec = spec.squeeze(0)
        spec = 20 * torch.log10(spec) - 20 * torch.log10(spec.max())
        return spec.T.float().cpu().numpy()

    def _feat(self, cqt: np.ndarray) -> torch.Tensor:
        feat = cqt[: self.chunk_frames]
        if len(feat) < self.chunk_frames:
            feat = np.pad(feat, ((0, self.chunk_frames - len(feat)), (0, 0)), constant_values=-100)
        n = len(feat) // self.mean_size
        feat = feat[: n * self.mean_size].reshape(n, self.mean_size, -1).mean(axis=1)
        return torch.from_numpy(feat.astype(np.float32))

    @torch.inference_mode()
    def _embed_feats(self, feats: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(feats).to(self.device)
        emb, _ = self._model.inference(batch)
        return emb.float().cpu().numpy()

    def embed_windows(self, item: AudioItem) -> np.ndarray:
        """Embeddings (n_fenêtres, 128) normalisés, fenêtres de `win_s` avec pas `hop_s`."""
        self._load()
        x = self._to_wave(item)
        cqt = self._cqt_db(x)
        win, hop = int(self.win_s * FPS), int(self.hop_s * FPS)
        starts = list(range(0, max(1, len(cqt) - win + 1), hop)) or [0]
        feats = [self._feat(cqt[s : s + win]) for s in starts]
        return l2_normalize(self._embed_feats(feats))

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        if self.window == "sliding":
            return np.stack([l2_normalize(self.embed_windows(it).mean(axis=0))[0] for it in items])
        feats = [self._feat(self._cqt_db(self._to_wave(it))) for it in items]
        return self._embed_feats(feats)

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        raise NotImplementedError("CoverHunter n'encode pas le texte")
