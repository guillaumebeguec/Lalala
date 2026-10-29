import pytest

from core import rerank as rr


def test_rerank_combines_scores(monkeypatch):
    def fake_chat(messages, json_schema=None, **k):
        ids = [line[5:-1] for line in messages[-1]["content"].splitlines() if line.startswith("[id: ")]
        class R:
            json = {"scores": [{"id": i, "score": 9 if i == "b" else 2, "reason": "era matches"} for i in ids]}
        return R()
    monkeypatch.setattr(rr.llm, "chat", fake_chat)
    cards = {"a": "Titre : A\nAnnée : 1999", "b": "Titre : B\nAnnée : 1985", "c": "Titre : C"}
    out = rr.rerank("80s song", [("a", 0.9), ("b", 0.8), ("c", 0.1)], cards, alpha=0.5, batch_size=2)
    assert out[0].id == "b" and out[0].llm_score == 9 and out[0].reason == "era matches"
    assert abs(out[0].final - (0.5 * (0.8 - 0.1) / 0.8 + 0.5 * 0.9)) < 1e-9


def test_rerank_without_llm_keeps_fusion_order(monkeypatch):
    monkeypatch.setattr(rr.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(rr.llm.LLMUnavailable("down")))
    out = rr.rerank("x", [("a", 0.9), ("b", 0.5)], {"a": "A", "b": "B"}, alpha=0.5)
    assert [r.id for r in out] == ["a", "b"] and out[0].llm_score is None


@pytest.mark.llm
def test_rerank_live_prefers_matching_card():
    cards = {
        "dz1": "Titre : Bette Davis Eyes\nArtiste : Kim Carnes\nAnnée : 1981\nGenre : Pop Rock\nLangue : english\n[généré] Description audio : synth-driven 80s pop with raspy female vocals",
        "dz2": "Titre : Master of Puppets\nArtiste : Metallica\nAnnée : 1986\nGenre : Thrash Metal\nLangue : english",
        "dz3": "Titre : La Bohème\nArtiste : Charles Aznavour\nAnnée : 1965\nGenre : Chanson\nLangue : french",
    }
    out = rr.rerank("80s pop song with a raspy female voice and synths, it was number one in the US", [("dz2", 0.9), ("dz1", 0.8), ("dz3", 0.7)], cards, alpha=0.3)
    assert out[0].id == "dz1"
