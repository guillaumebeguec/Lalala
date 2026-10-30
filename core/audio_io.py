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


def sniff_suffix(data: bytes) -> str:
    """Extension probable d'après les octets de tête (webm/opus, mp4/m4a, ogg, wav, flac, mp3)."""
    head = data[:16]
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"RIFF"):
        return ".wav"
    if head.startswith(b"fLaC"):
        return ".flac"
    return ".mp3"


def _decode_with_ffmpeg(src: Path | str, sr: int | None, mono: bool) -> tuple[np.ndarray, int]:
    """Décodage par ffmpeg (webm/opus, m4a/aac, mp4, mp3 abîmé…) vers PCM float32."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg introuvable (brew install ffmpeg)")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(src), "-f", "f32le", "-acodec", "pcm_f32le"]
    if mono:
        cmd += ["-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd.append("-")
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"ffmpeg : {out.stderr.decode(errors='ignore')[:200]}")
    x = np.frombuffer(out.stdout, dtype=np.float32)
    if not sr:
        # fréquence native : relue via ffprobe
        ffprobe = shutil.which("ffprobe")
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                                "-of", "csv=p=0", str(src)], capture_output=True, text=True) if ffprobe else None
        sr = int(probe.stdout.strip().splitlines()[0]) if probe and probe.stdout.strip() else 48_000
    return x.copy(), int(sr)


def load_audio(src: bytes | str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Charge un fichier ou des octets (mp3, wav, webm/opus, m4a, mp4…). Renvoie (signal, sr).

    libsndfile d'abord (wav, flac, mp3) ; sinon ffmpeg (enregistrements du navigateur en
    webm/opus ou m4a/aac, vidéos)."""
    import soundfile as sf
    is_bytes = isinstance(src, (bytes, bytearray))
    suffix = sniff_suffix(bytes(src[:16])) if is_bytes else Path(str(src)).suffix.lower()
    if suffix in (".wav", ".flac", ".mp3", ".ogg"):
        try:
            f = io.BytesIO(src) if is_bytes else str(src)
            x, native_sr = sf.read(f, dtype="float32", always_2d=False)
            if mono:
                x = to_mono(x)
            if sr is not None:
                x = resample(x, native_sr, sr)
                native_sr = sr
            return x.astype(np.float32), int(native_sr)
        except Exception as e:  # noqa: BLE001 — mp3 abîmé, codec non géré : on passe à ffmpeg
            logger.debug("soundfile a échoué (%s), repli ffmpeg", e)
    if is_bytes:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(src)
            tmp.flush()
            x, native_sr = _decode_with_ffmpeg(tmp.name, sr, mono)
    else:
        x, native_sr = _decode_with_ffmpeg(src, sr, mono)
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
