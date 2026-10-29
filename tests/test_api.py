from fastapi.testclient import TestClient

from api import main as api
from finder.agent import Event


class FakeAgent:
    def __init__(self):
        from finder.agent import SessionState
        self.state = SessionState()

    def step(self, text=None, audio=None):
        yield Event("clue", {"text": text or "audio"})
        yield Event("question", {"question": "Décennie ?", "options": ["1980s", "1990s"]})


def test_message_sse_and_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "new_agent", lambda: FakeAgent())
    monkeypatch.setattr(api, "replays_dir", lambda: tmp_path)
    c = TestClient(api.app)
    sid = c.post("/session").json()["session_id"]
    r = c.post(f"/session/{sid}/message", data={"text": "une chanson des années 80"})
    assert r.status_code == 200 and "event: question" in r.text and "event: end" in r.text
    assert c.get(f"/session/{sid}/state").json()["done"] is False
    assert c.post(f"/session/{sid}/save/demo1").json()["events"] == 2
    assert c.get("/replays").json() == ["demo1"]
    rr = c.post("/replay/demo1?delay=0")
    assert "event: clue" in rr.text and "event: end" in rr.text
    assert c.post("/replay/nope").status_code == 404
    assert c.get("/health").json()["ok"] is True
