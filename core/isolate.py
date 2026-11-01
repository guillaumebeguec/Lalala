"""Isolation de la musique avec SAM Audio (Meta), phase 9.

SAM Audio sépare une source décrite en texte (« music ») du reste (voix, bruit de salle). On
l'applique aux enregistrements bruités avant l'empreinte et la similarité audio. Le modèle tourne
dans un **sous-processus** avec son propre venv (voir `scripts/sam_runner*.py`) : le paquet officiel
dépend de `perception_models`, qui installe un paquet Python `core` en conflit avec le nôtre, et le
portage MLX a ses propres dépendances.

Moteurs, par ordre de préférence (`SAM_AUDIO_ENGINE=auto|mlx|torch|off`) :
- `mlx`   : `data/raw/sam_audio/venv_mlx` + `mlx-community/sam-audio-base` (Apple Silicon natif) ;
- `torch` : `data/raw/sam_audio/venv` + `facebook/sam-audio-base` (poids officiels, accès restreint, HF_TOKEN).

Résultats mis en cache par hash du contenu (`data/cache/sam_audio/`). Toute erreur → `None` :
l'appelant continue avec l'audio d'origine.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from core.audio_io import load_audio, to_wav_bytes
from core.cache import DiskCache
from core.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

SAM_DIR = PROJECT_ROOT / "data" / "raw" / "sam_audio"
ENGINES = {
    "mlx": (SAM_DIR / "venv_mlx" / "bin" / "python", PROJECT_ROOT / "scripts" / "sam_runner_mlx.py"),
    "torch": (SAM_DIR / "venv" / "bin" / "python", PROJECT_ROOT / "scripts" / "sam_runner.py"),
}


def engine() -> str | None:
    """Moteur disponible selon `SAM_AUDIO_ENGINE` et les venvs présents ; None si désactivé/absent."""
    get_settings()  # charge .env (HF_TOKEN, SAM_AUDIO_ENGINE)
    want = os.environ.get("SAM_AUDIO_ENGINE", "auto").lower()
    if want in ("off", "0", "false", "none"):
        return None
    order = ["mlx", "torch"] if want == "auto" else [want]
    for name in order:
        py, runner = ENGINES.get(name, (None, None))
        if py and py.exists() and runner.exists():
            return name
    return None


def isolate_music(audio: bytes | tuple[np.ndarray, int], prompt: str = "music", timeout: float = 600.0,
                  sr_out: int = 16_000) -> tuple[bytes, dict] | None:
    """Renvoie (wav mono `sr_out` de la musique isolée, infos {engine, seconds, sr}) ou None."""
    name = engine()
    if name is None:
        return None
    if isinstance(audio, (bytes, bytearray)):
        x, sr = load_audio(bytes(audio), sr=None)
    else:
        x, sr = audio
    wav_in = to_wav_bytes(np.asarray(x, dtype=np.float32), int(sr))
    cache = DiskCache("sam_audio")
    key = cache.key(wav_in, engine=name, prompt=prompt, v=1)
    cached = cache.get_bytes(key, ext=".wav")
    info = cache.get(key + "_info") or {}
    if cached:
        return cached, {**info, "cached": True}

    py, runner = ENGINES[name]
    with tempfile.TemporaryDirectory(prefix="sam_") as tmp:   # hors du projet : évite le conflit de paquet `core`
        tmp = Path(tmp)
        (tmp / "in.wav").write_bytes(wav_in)
        cmd = [str(py), str(runner), "--in", str(tmp / "in.wav"), "--out", str(tmp / "out.wav"), "--prompt", prompt]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("SAM Audio (%s) : délai dépassé (%.0f s)", name, timeout)
            return None
        if proc.returncode != 0 or not (tmp / "out.wav").exists():
            logger.warning("SAM Audio (%s) : échec — %s", name, (proc.stderr or "").strip()[-400:])
            return None
        try:
            info = json.loads([l for l in proc.stdout.splitlines() if l.startswith("{")][-1])
        except (ValueError, IndexError):
            info = {}
        y, sr_y = load_audio(tmp / "out.wav", sr=sr_out)
    out = to_wav_bytes(y.astype(np.float32), sr_y)
    info = {"engine": name, "seconds": round(time.time() - t0, 1), "sr": sr_y, "model": info.get("model")}
    cache.put_bytes(key, out, ext=".wav")
    cache.put(key + "_info", info)
    logger.info("SAM Audio (%s) : %.1f s pour %.1f s d'audio", name, info["seconds"], len(x) / sr)
    return out, info
