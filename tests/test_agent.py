"""Boucle de l'agent avec outils et LLM simulés : garde-fous, événements, pas de boucle infinie."""
import json

from finder import agent as ag
from finder.tools import Candidate


class FakeTools:
    def __init__(self):
        self.calls = []
        self._catalog = type("C", (), {"__len__": lambda s: 3})()
        self.catalog = self._catalog

    def catalog_search(self, text, hints=None):
        self.calls.append(("catalog", text))
        return {"candidates": [Candidate("dz1", "Bette Davis Eyes", "Kim Carnes", 1981, 0.9, "catalog", "80s synth pop"),
                               Candidate("dz2", "Sweet Dreams", "Eurythmics", 1983, 0.7, "catalog")], "rewrite": None, "lists": {}}

    def web_search(self, q, k=6):
        self.calls.append(("web", q)); return []

    def audio_match(self, audio):
        return {"candidates": [], "exact": None}


def _tc(name, args, i=1):
    return ag.llm.ToolCall(id=f"c{i}", name=name, arguments=args)


def _resp(tool_calls, content=""):
    r = ag.llm.ChatResult(content=content, tool_calls=tool_calls)
    return r


def test_search_then_ask_then_propose(monkeypatch):
    script = [
        _resp([_tc("catalog_search", {"text": "80s song female voice"})]),
        _resp([_tc("ask_user", {"question": "Voix ?", "options": ["féminine", "masculine"]}, 2)]),
        _resp([_tc("propose", {"candidates": [{"track_id": "dz1", "confidence": 0.8, "reason": "ça colle"}, {"track_id": "zzz", "confidence": 0.5, "reason": "inventé"}]}, 3)]),
    ]
    monkeypatch.setattr(ag.llm, "chat", lambda *a, **k: script.pop(0))
    a = ag.Agent(FakeTools(), max_questions=3, max_tool_calls=8)
    ev = list(a.step("Chanson des années 80 avec une voix de femme"))
    types = [e.type for e in ev]
    assert types[0] == "clue" and "search" in types and types[-1] == "question"
    assert a.state.pending_question["options"][-1] == ag.DONT_KNOW
    ev2 = list(a.step("féminine"))
    assert ev2[0].type == "clue" and ev2[-1].type == "proposal"
    props = ev2[-1].data["candidates"]
    assert [p["track_id"] for p in props] == ["dz1"]  # « zzz » rejeté : pas issu d'une recherche
    assert a.state.done and a.state.tool_calls == 3


def test_tool_call_budget_forces_proposal(monkeypatch):
    monkeypatch.setattr(ag.llm, "chat", lambda *a, **k: _resp([_tc("catalog_search", {"text": "x"})]))
    tools = FakeTools()
    a = ag.Agent(tools, max_tool_calls=4)
    ev = list(a.step("x"))
    assert ev[-1].type == "proposal" and a.state.tool_calls <= 4
    assert len([c for c in tools.calls if c[0] == "catalog"]) == 1  # requête répétée refusée


def test_llm_down_still_proposes(monkeypatch):
    def boom(*a, **k):
        raise ag.llm.LLMUnavailable("down")
    monkeypatch.setattr(ag.llm, "chat", boom)
    a = ag.Agent(FakeTools())
    ev = list(a.step("x"))
    assert any(e.type == "error" for e in ev) and ev[-1].type == "message"


def test_no_rejects_candidates(monkeypatch):
    script = [_resp([_tc("catalog_search", {"text": "a"})]), _resp([_tc("propose", {"candidates": [{"track_id": "dz1", "confidence": 0.9, "reason": "r"}]}, 2)]),
              _resp([_tc("catalog_search", {"text": "b"})]), _resp([_tc("propose", {"candidates": [{"track_id": "dz2", "confidence": 0.6, "reason": "r"}]}, 4)])]
    monkeypatch.setattr(ag.llm, "chat", lambda *a, **k: script.pop(0))
    a = ag.Agent(FakeTools())
    ev = list(a.step("desc"))
    assert ev[-1].data["candidates"][0]["track_id"] == "dz1"
    a.state.done = False
    ev2 = list(a.step("non"))
    assert "dz1" in a.state.rejected and ev2[-1].data["candidates"][0]["track_id"] == "dz2"
