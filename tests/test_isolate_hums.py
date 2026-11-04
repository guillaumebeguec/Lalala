"""core.isolate (SAM Audio engine selection) and testsets.hums (T6b) without a model or network."""

import numpy as np
import pandas as pd

from core import isolate
from testsets import hums


def test_engine_off(monkeypatch):
    monkeypatch.setenv("SAM_AUDIO_ENGINE", "off")
    assert isolate.engine() is None
    assert isolate.isolate_music(b"") is None


def test_engine_missing_venv(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM_AUDIO_ENGINE", "auto")
    monkeypatch.setattr(isolate, "ENGINES", {"mlx": (tmp_path / "nope", tmp_path / "nope.py")})
    assert isolate.engine() is None


def test_manual_hums_rows(monkeypatch, tmp_path):
    class S:
        manual_dir = tmp_path
    monkeypatch.setattr(hums, "get_settings", lambda: S())
    (tmp_path / "hums").mkdir()
    (tmp_path / "hums_synth").mkdir()
    (tmp_path / "hums" / "dz1_1.m4a").write_bytes(b"x")
    (tmp_path / "hums" / "dz1_2.wav").write_bytes(b"x")
    (tmp_path / "hums" / ".DS_Store").write_bytes(b"x")
    (tmp_path / "hums_synth" / "dz2_1.wav").write_bytes(b"x")
    (tmp_path / "hums_synth" / "dz9_1.wav").write_bytes(b"x")
    rows = hums.manual_hums({"dz1", "dz2"})
    assert [r["reponse_id"] for r in rows] == ["dz1", "dz1", "dz2"]
    assert {r["kind"] for r in rows} == {"real", "synth"}
    assert all(r["id"].startswith("t6b_") for r in rows)
