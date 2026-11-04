"""Melody matching (core.melody) on synthetic contours: transposition, tempo, gaps."""

import numpy as np
import pytest

from core.melody import ContourStore, _resample_tempo, contour_from_track, match


def _melody(rng, n_notes=12, note_len=25):
    """Contour in MIDI semitones: held notes (20 ms frames), short unvoiced pauses."""
    notes = 60 + rng.integers(-7, 8, size=n_notes)
    parts = []
    for v in notes:
        parts.append(np.full(note_len, float(v), np.float32))
        parts.append(np.full(3, np.nan, np.float32))
    return np.concatenate(parts)


def _store(rng, n=60, length=1500):
    contours = {}
    for i in range(n):
        c = np.full(length, np.nan, np.float32)
        pos = 0
        while pos < length:
            m = _melody(rng, n_notes=rng.integers(6, 14))
            c[pos : pos + len(m)] = m[: max(0, length - pos)]
            pos += len(m) + rng.integers(10, 60)
        contours[f"t{i}"] = c
    return ContourStore.build(contours)


def test_contour_from_track_voicing_and_frames():
    f0 = np.array([440.0, 440.0, 0.0, 0.0, 220.0, 220.0], np.float32)
    conf = np.array([0.9, 0.9, 0.0, 0.0, 0.8, 0.1], np.float32)
    c = contour_from_track(np.stack([f0, conf]))
    assert len(c) == 3
    assert c[0] == pytest.approx(69.0)          # A 440 Hz = MIDI 69
    assert np.isnan(c[1])                        # unvoiced
    assert c[2] == pytest.approx(57.0)           # one voiced frame out of two is enough


def test_resample_tempo_lengths():
    q = np.arange(100, dtype=np.float32)
    assert len(_resample_tempo(q, 1.0)) == 100
    assert len(_resample_tempo(q, 0.8)) == 80
    assert len(_resample_tempo(q, 1.25)) == 125


def test_match_finds_transposed_and_slowed_fragment():
    rng = np.random.default_rng(0)
    store = _store(rng)
    target = store.ids[17]
    src = store.data[17]
    start = 400
    frag = src[start : start + 500].copy()
    # the person sings 3 semitones lower, 10% slower, with some blur and gaps
    q = _resample_tempo(frag - 3.0, 1.1)
    q = q + rng.normal(0, 0.3, len(q)).astype(np.float32)
    q[100:130] = np.nan
    res = match(q, store, top=5)
    assert res and res[0][0] == target
    assert res[0][1] > 0.7
    assert abs(res[0][2] - start * 0.02) < 1.0   # position recovered within ± 1 s


def test_match_rejects_unvoiced_query():
    rng = np.random.default_rng(1)
    store = _store(rng, n=5)
    q = np.full(300, np.nan, np.float32)
    assert match(q, store) == []


def test_store_roundtrip(tmp_path, monkeypatch):
    from core import melody
    monkeypatch.setattr(melody.ContourStore, "path", staticmethod(lambda subset: tmp_path / f"contours_{subset}.npz"))
    rng = np.random.default_rng(2)
    st = _store(rng, n=3, length=200)
    st.save("x")
    st2 = melody.ContourStore.load("x")
    assert st2.ids == st.ids and st2.data.shape == st.data.shape
    assert np.array_equal(np.isnan(st2.data), np.isnan(st.data))


def test_merged_store_keeps_best_window_per_track(tmp_path, monkeypatch):
    from core import melody
    monkeypatch.setattr(melody.ContourStore, "path", staticmethod(lambda subset, provider=None: tmp_path / f"contours_{subset}{'_' + provider if provider else ''}.npz"))
    rng = np.random.default_rng(3)
    st = _store(rng, n=10, length=600)
    st.save("y")
    # iTunes window: track t3 with another passage, which contains the query
    extra = {"t3": st.data[7].copy()}
    np.savez_compressed(melody.ContourStore.path("y", "itunes"), ids=np.asarray(list(extra)), data=np.stack(list(extra.values())))
    merged = melody.ContourStore.load("y")
    assert len(merged) == 11 and merged.ids.count("t3") == 2
    q = st.data[7][100:400].copy() + 2.0
    res = match(q, merged, top=5)
    ids = [r[0] for r in res]
    assert len(ids) == len(set(ids))            # a single result per track
    assert set(ids[:2]) == {"t3", "t7"}         # the two identical windows come first
