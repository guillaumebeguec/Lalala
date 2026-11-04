import shutil

import numpy as np
import pytest

from core.fingerprint import FingerprintIndex, fpcalc_raw, match_score

pytestmark = pytest.mark.skipif(shutil.which("fpcalc") is None, reason="fpcalc not installed")


def _signal(seed, sr=16000, dur=12.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    freqs = rng.uniform(100, 2000, size=12)
    x = sum(np.sin(2 * np.pi * f * t) * (1 + np.sin(2 * np.pi * 0.5 * t * (i + 1))) for i, f in enumerate(freqs))
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32), sr


def test_fingerprint_self_and_shifted_match_beats_other():
    a, sr = _signal(1)
    b, _ = _signal(2)
    fa, fb = fpcalc_raw((a, sr)), fpcalc_raw((b, sr))
    assert len(fa) > 40 and fa.dtype == np.uint32
    assert match_score(fa, fa)[0] > 0.99
    shifted = fpcalc_raw((a[sr * 3 :], sr))  # same audio, 3 s later
    s_same, off = match_score(shifted, fa)
    s_other, _ = match_score(shifted, fb)
    assert s_same > 0.85 and s_same > s_other + 0.2
    assert 18 <= off <= 30  # ≈ 3 s × 8 values/s
    idx = FingerprintIndex({"a": fa, "b": fb})
    assert idx.search(shifted, k=1)[0][0] == "a"
    idx.save()
    assert FingerprintIndex.load().search(shifted, k=1)[0][0] == "a"
