import numpy as np

from core.audio_io import crop, load_audio, mix_at_snr, random_window, resample, rms, to_wav_bytes


def _tone(sr=16000, f=440.0, dur=2.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_wav_roundtrip_and_resample():
    sr = 16000
    x = _tone(sr)
    y, sr2 = load_audio(to_wav_bytes(x, sr))
    assert sr2 == sr and y.shape == x.shape and np.abs(y - x).max() < 1e-3
    z, sr3 = load_audio(to_wav_bytes(x, sr), sr=8000)
    assert sr3 == 8000 and abs(len(z) - len(x) / 2) <= 2
    assert resample(x, sr, sr) is x


def test_mix_at_snr_hits_target():
    sr = 16000
    rng = np.random.default_rng(0)
    sig = _tone(sr, amp=0.1)
    noise = rng.normal(0, 0.1, size=sr * 5).astype(np.float32)
    for snr in (0, 5, 10):
        mixed = mix_at_snr(sig, noise, snr, rng)
        added = mixed - sig
        assert np.max(np.abs(mixed)) < 1.0  # no peak renormalisation at these amplitudes
        assert abs(20 * np.log10(rms(sig) / rms(added)) - snr) < 0.1
        assert mixed.dtype == np.float32 and len(mixed) == len(sig)


def test_mix_short_noise_is_tiled_and_clipping_normalized():
    sig = _tone(amp=0.9)
    noise = np.ones(100, dtype=np.float32)
    mixed = mix_at_snr(sig, noise, -10)
    assert len(mixed) == len(sig) and np.max(np.abs(mixed)) <= 1.0


def test_windows():
    sr = 1000
    x = np.arange(5000, dtype=np.float32)
    seg, start = random_window(x, sr, 1.0, np.random.default_rng(1))
    assert len(seg) == 1000 and seg[0] == start * sr
    assert len(crop(x, sr, 4.5, 1.0)) == 1000  # padding
    assert len(crop(x, sr, 4.5, 1.0, pad=False)) == 500
