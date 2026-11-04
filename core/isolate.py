"""Music isolation with SAM Audio (Meta), phase 9.

SAM Audio separates a source described in text ("music") from the rest (voice, room noise). We
apply it to noisy recordings before fingerprinting and audio similarity. The model runs
in a **subprocess** with its own venv (see `scripts/sam_runner*.py`): the official package
depends on `perception_models`, which installs a Python package named `core` that conflicts with ours, and the
MLX port has its own dependencies.

Engines, in order of preference (`SAM_AUDIO_ENGINE=auto|mlx|torch|off`):
- `mlx`  : `data/raw/sam_audio/venv_mlx` + `mlx-community/sam-audio-base` (native Apple Silicon);
- `torch`: `data/raw/sam_audio/venv` + `facebook/sam-audio-base` (official weights, gated access, HF_TOKEN).

Results cached by content hash (`data/cache/sam_audio/`). Any error → `None`:
the caller continues with the original audio.
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
    """Available engine according to `SAM_AUDIO_ENGINE` and the present venvs; None if disabled/absent."""
    get_settings()  # loads .env (HF_TOKEN, SAM_AUDIO_ENGINE)
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
    """Return (mono wav at `sr_out` of the isolated music, info {engine, seconds, sr}) or None."""
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
    with tempfile.TemporaryDirectory(prefix="sam_") as tmp:   # outside the project: avoids the `core` package conflict
        tmp = Path(tmp)
        (tmp / "in.wav").write_bytes(wav_in)
        cmd = [str(py), str(runner), "--in", str(tmp / "in.wav"), "--out", str(tmp / "out.wav"), "--prompt", prompt]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("SAM Audio (%s): timeout exceeded (%.0f s)", name, timeout)
            return None
        if proc.returncode != 0 or not (tmp / "out.wav").exists():
            logger.warning("SAM Audio (%s): failure — %s", name, (proc.stderr or "").strip()[-400:])
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
    logger.info("SAM Audio (%s): %.1f s for %.1f s of audio", name, info["seconds"], len(x) / sr)
    return out, info
