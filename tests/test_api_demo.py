import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import demo as demo_api
from api import main as api
from finder.tools import Candidate


class FakeTB:
    def __init__(self):
        self.catalog = pd.DataFrame({"track_id": ["dz1", "dz2"], "album_title": ["Alb", None], "deezer_id": [1, 2]}).set_index("track_id", drop=False)
        self.previews = pd.DataFrame({"track_id": ["dz1"], "preview_url": ["http://x/1.mp3"]}).set_index("track_id")

    def active_embedders(self):
        return ["clap", "gemma"]

    def catalog_search(self, text, hints=None):
        return {"candidates": [Candidate("dz1", "Bette Davis Eyes", "Kim Carnes", 1981, 0.9, "catalog", "era matches", "http://x/1.mp3"),
                               Candidate("dz2", "Sweet Dreams", "Eurythmics", 1983, 0.4, "catalog")], "rewrite": None, "lists": {"bm25:brut": 50}}

    def audio_match(self, audio):
        return {"candidates": [Candidate("dz1", "Bette Davis Eyes", "Kim Carnes", 1981, 0.7, "catalog")], "exact": None}


def _events(text):
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev = [l for l in block.split("\n") if l.startswith("event:")][0][7:]
        data = [l for l in block.split("\n") if l.startswith("data:")][0][6:]
        out.append((ev, json.loads(data)))
    return out


def test_text_search_stream_and_feedback(tmp_path, monkeypatch):
    demo_api.set_toolbox(FakeTB())
    monkeypatch.setattr(demo_api, "replays_dir", lambda: tmp_path)
    c = TestClient(api.app)
    r = c.post("/api/search/stream", data={"text": "80s song raspy female voice"})
    assert r.status_code == 200
    ev = _events(r.text)
    kinds = [e for e, _ in ev]
    assert kinds[0] == "input" and kinds[-1] == "result" and "step" in kinds
    res = ev[-1][1]
    assert res["best"]["title"] == "Bette Davis Eyes" and res["confidence"] == "high"
    assert res["best"]["preview_url"] == "/api/preview/dz1" and res["alternatives"][0]["id"] == "dz2"
    assert any(p["source"] == "Context" for p in res["proofs"])
    titles = {d["title"]: d["status"] for e, d in ev if e == "step"}
    assert titles["Ranking candidates"] == "done" and titles["Verification"] == "skipped"
    assert c.post("/api/feedback", json={"query_id": res["query_id"], "track_id": "dz1", "verdict": "match"}).json()["ok"]
    assert c.post(f"/api/save/{res['query_id']}/memory").json()["scene"] == "memory"
    assert c.get("/api/replays").json() == ["memory"]
    assert [e for e, _ in _events(c.get("/api/replay/memory?delay=0").text)][-1] == "result"
    assert c.post("/api/search", data={"text": "x"}).json()["best"]["id"] == "dz1"
    assert c.post("/api/search/stream", data={}).status_code == 400


def test_audio_search_stream():
    from core.audio_io import to_wav_bytes
    demo_api.set_toolbox(FakeTB())
    c = TestClient(api.app)
    sr = 16000
    wav = to_wav_bytes((0.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr)).astype("float32"), sr)
    r = c.post("/api/search/stream", data={"text": "piano"}, files={"audio": ("rec.wav", wav, "audio/wav")})
    ev = _events(r.text)
    inp = ev[0][1]
    assert inp["kind"] == "mixed" and len(inp["waveform"]) == 64 and inp["duration_s"] == 3.0 and inp["note"] == "piano"
    steps = [d for e, d in ev if e == "step"]
    assert steps[0]["title"] == "Isolating the music"
    assert ev[-1][0] == "result" and ev[-1][1]["best"]["id"] == "dz1"
