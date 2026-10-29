"""Chargement, rééchantillonnage, découpage et mixage audio.

Tout l'audio circule en `np.ndarray` float32 mono, normalisé dans [-1, 1], avec
sa fréquence d'échantillonnage. Les MP3 peuvent venir d'octets en mémoire
(mode `PREVIEW_STORAGE=memory`) : on passe par `soundfile` (libsndfile ≥ 1.1
lit le MP3) avec repli `librosa`/`audioread`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SR = 48_000


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    import librosa
    return librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


def load_audio(src: bytes | str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Charge un fichier ou des octets (mp3, wav, m4a…). Renvoie (signal, sr)."""
    import soundfile as sf
    try:
        f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)
        x, native_sr = sf.read(f, dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001 — libsndfile sans mp3/m4a : repli audioread via librosa
        logger.debug("soundfile a échoué (%s), repli librosa", e)
        import librosa
        if isinstance(src, (bytes, bytearray)):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                tmp.write(src)
                tmp.flush()
                x, native_sr = librosa.load(tmp.name, sr=None, mono=False)
        else:
            x, native_sr = librosa.load(str(src), sr=None, mono=False)
        x = x.T if x.ndim == 2 else x
    if mono:
        x = to_mono(x)
    if sr is not None:
        x = resample(x, native_sr, sr)
        native_sr = sr
    return x.astype(np.float32), int(native_sr)


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) if len(x) else 0.0


def crop(x: np.ndarray, sr: int, start_s: float, duration_s: float, pad: bool = True) -> np.ndarray:
    a = int(round(start_s * sr))
    n = int(round(duration_s * sr))
    seg = x[a : a + n]
    if pad and len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg


def random_window(x: np.ndarray, sr: int, duration_s: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Fenêtre de `duration_s` à position aléatoire ; renvoie (segment, début en s)."""
    n = int(round(duration_s * sr))
    if len(x) <= n:
        return crop(x, sr, 0.0, duration_s), 0.0
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start : start + n], start / sr


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mélange `signal` et `noise` (même sr) au rapport signal/bruit demandé, en dB,
    mesuré en RMS. Le bruit est bouclé ou découpé à la longueur du signal ; le
    résultat est ramené dans [-1, 1] si nécessaire."""
    n = len(signal)
    if len(noise) == 0:
        return signal.copy()
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > n:
        start = int(rng.integers(0, len(noise) - n + 1)) if rng is not None else 0
        noise = noise[start : start + n]
    s_rms, n_rms = rms(signal), rms(noise)
    if s_rms == 0 or n_rms == 0:
        return signal.copy()
    gain = s_rms / (n_rms * 10 ** (snr_db / 20))
    out = signal + gain * noise
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def measured_snr_db(signal: np.ndarray, noise_scaled: np.ndarray) -> float:
    return 20 * np.log10(rms(signal) / max(rms(noise_scaled), 1e-12))
