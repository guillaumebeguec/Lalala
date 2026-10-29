"""T3 — Extraits bruités (PLAN.md phase 2.3).

100 extraits du catalogue (avec extrait disponible, hors ToT) mixés avec du bruit
MUSAN (`noise/`) et du brouhaha (somme de 3 à 5 voix de `speech/`) à SNR 0, 5 et
10 dB, fenêtres de 10 s à position aléatoire → 300 cas + les vrais
enregistrements de `data/manual/real_recordings/<track_id>.m4a`.

Comme l'audio du catalogue n'est pas conservé sur disque (PREVIEW_STORAGE=memory),
le jeu est figé sous forme de **recette** (data/testsets/t3_noisy.jsonl) :
track_id, début de fenêtre, fichiers MUSAN et décalages, SNR, graine. `render`
reconstruit le mélange à l'identique à l'évaluation. Les bruits MUSAN sont
CC BY / domaine public (citer Snyder et al., 2015).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from core.audio_io import crop, load_audio, mix_at_snr, random_window, rms
from core.config import get_settings
from testsets.common import SEED, testsets_dir, write_jsonl

logger = logging.getLogger(__name__)

SNRS = (0, 5, 10)
WINDOW_S = 10.0
SR = 32_000


def musan_dir() -> Path:
    return get_settings().data_dir / "raw" / "musan" / "musan"


def _list_wavs(sub: str, min_duration_s: float = 12.0) -> list[Path]:
    import soundfile as sf
    out = []
    for p in sorted((musan_dir() / sub).rglob("*.wav")):
        try:
            info = sf.info(str(p))
        except RuntimeError:
            continue
        if info.duration >= min_duration_s:
            out.append(p)
    return out


def build_t3_recipe(catalog: pd.DataFrame, previews: pd.DataFrame, n_tracks: int = 100, seed: int = SEED) -> list[dict]:
    ok = set(previews[previews.status == "ok"].track_id)
    pool = catalog[catalog.track_id.isin(ok) & ~catalog.is_tot_answer]
    rng = random.Random(seed)
    tracks = pool.sample(n=min(n_tracks, len(pool)), random_state=seed)
    noises = _list_wavs("noise")
    speeches = _list_wavs("speech")
    if not noises or not speeches:
        raise FileNotFoundError(f"MUSAN introuvable ou vide dans {musan_dir()}")
    rows = []
    for i, r in enumerate(tracks.itertuples(index=False)):
        kind = "noise" if i % 2 == 0 else "babble"
        window_start = round(rng.uniform(0.0, 30.0 - WINDOW_S), 2)
        if kind == "noise":
            parts = [{"file": str(rng.choice(noises).relative_to(musan_dir())), "offset_s": round(rng.uniform(0, 60), 2)}]
        else:
            parts = [{"file": str(rng.choice(speeches).relative_to(musan_dir())), "offset_s": round(rng.uniform(0, 120), 2)}
                     for _ in range(rng.randint(3, 5))]
        for snr in SNRS:
            rows.append({"id": f"t3_{r.track_id}_{snr}", "reponse_id": r.track_id, "kind": kind, "snr_db": snr,
                         "window_start_s": window_start, "noise_parts": parts, "seed": seed + len(rows)})
    logger.info("T3 : %d cas synthétiques (%d morceaux × %d SNR)", len(rows), len(tracks), len(SNRS))
    return rows


def _load_part(part: dict, duration_s: float, sr: int) -> np.ndarray:
    x, _ = load_audio(musan_dir() / part["file"], sr=sr)
    seg = crop(x, sr, part["offset_s"], duration_s, pad=False)
    if len(seg) < int(duration_s * sr):  # fichier trop court après l'offset : on boucle depuis le début
        reps = int(np.ceil(duration_s * sr / max(len(x), 1))) + 1
        seg = crop(np.tile(x, reps), sr, part["offset_s"] % (len(x) / sr), duration_s)
    return seg


def render(case: dict, clean_mp3: bytes, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruit (mélange, signal propre) pour un cas de la recette."""
    x, _ = load_audio(clean_mp3, sr=sr)
    clean = crop(x, sr, case["window_start_s"], WINDOW_S)
    parts = [_load_part(p, WINDOW_S, sr) for p in case["noise_parts"]]
    noise = parts[0] if len(parts) == 1 else sum(p / max(rms(p), 1e-6) for p in parts)
    mixed = mix_at_snr(clean, noise.astype(np.float32), case["snr_db"], np.random.default_rng(case["seed"]))
    return mixed, clean


def real_recordings() -> list[dict]:
    d = get_settings().manual_dir / "real_recordings"
    rows = []
    for p in sorted(d.glob("*.m4a")) + sorted(d.glob("*.mp3")) + sorted(d.glob("*.wav")):
        rows.append({"id": f"t3_real_{p.stem}", "reponse_id": p.stem, "kind": "real", "snr_db": None, "file": str(p)})
    return rows


def write_t3(catalog: pd.DataFrame, previews: pd.DataFrame, **kw) -> Path:
    rows = build_t3_recipe(catalog, previews, **kw) + real_recordings()
    out = testsets_dir() / "t3_noisy.jsonl"
    write_jsonl(out, rows)
    return out
