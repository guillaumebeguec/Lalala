import pytest

from core import query_rewrite as qr


def test_strip_unsupported_names():
    desc = "Une pub Citroën vers 2008 avec un sifflement, chantée par un homme"
    q = "whistling pop song Citroën ad 2008 male vocals Young Folks Peter Bjorn and John"
    out = qr._strip_unsupported_names(q, desc, ["Citroën"])
    assert "Citroën" in out and "Young Folks" not in out and "Peter Bjorn" not in out


def test_rewrite_with_mocked_llm(monkeypatch):
    payload = {"captions": ["whistled melody over acoustic guitar", "mid-tempo indie pop", "male vocals"],
               "hints": {"decade": "2000s", "language": "english", "medium": "ad", "genre": "indie pop", "vocals": "male",
                         "lyrics_fragments": [], "named_entities": ["Citroën", "Coldplay"]},
               "query": "whistling indie pop Citroën ad 2008 Coldplay", "query_fr": "sifflement pop pub Citroën 2008"}

    class R:
        json = payload

    monkeypatch.setattr(qr.llm, "chat", lambda *a, **k: R())
    r = qr.rewrite("Une pub Citroën vers 2008 avec un sifflement, chantée par un homme", use_cache=False)
    assert r.decade == "2000s" and r.language is None      # "english" is not mentioned: hint dropped
    assert r.hints["medium"] == "ad" and r.hints["vocals"] == "male" and r.hints["genre"] is None
    assert r.hints["named_entities"] == ["Citroën"]       # Coldplay is not mentioned by the user
    assert "Coldplay" not in r.query and "Citroën" in r.query
    assert r.captions == ["whistled melody over acoustic guitar", "male vocals"]   # "indie pop" and the tempo are not mentioned
    assert r.query_fr.startswith("sifflement")


@pytest.mark.llm
def test_rewrite_live_does_not_invent_titles():
    desc = ("Chanson des années 80, un homme chante, il y a un gros solo de saxophone au milieu, "
            "c'était dans une pub pour une voiture, je crois que ça parle de la nuit.")
    r = qr.rewrite(desc, use_cache=False)
    assert 3 <= len(r.captions) <= 5
    assert r.decade == "1980s"
    assert r.hints.get("medium") and "ad" in str(r.hints["medium"]).lower()
    assert r.hints["named_entities"] == []
    assert "sax" in (r.query + " ".join(r.captions)).lower()


def test_ground_hints_drops_invented_genre_and_vocals():
    from core.query_rewrite import ground_hints
    h = ground_hints({"genre": "rock", "vocals": "male", "decade": "1990s", "sound_terms": ["guitare"]}, "guitare?")
    assert h["genre"] is None and h["vocals"] is None and h["decade"] is None
    assert h["sound_terms"] == ["guitare"]


def test_ground_hints_keeps_stated_clues():
    from core.query_rewrite import ground_hints
    h = ground_hints({"genre": "rock", "vocals": "female", "decade": "1990s", "language": "french", "medium": "ad"},
                     "une chanteuse, pub de voiture années 90, un peu rock, en français")
    assert h == {"genre": "rock", "vocals": "female", "decade": "1990s", "language": "french", "medium": "ad"}


def test_ground_captions_filters_added_genre_tempo_vocals():
    from core.query_rewrite import ground_captions
    caps = ground_captions(["Rock music with guitar", "Upbeat tempo", "Male vocals with guitar", "Song with a guitar"], "guitare?")
    assert caps == ["Song with a guitar"]
    assert ground_captions(["Rock music with guitar"], "guitare?") == ["guitare?"]


def test_sound_only_routing_fields():
    r = qr.Rewrite(captions=["Guitar sound"], hints={"sound_terms": ["guitar"]}, query="guitar",
                   raw={"sound_query": "guitar", "context_query": ""})
    assert r.sound_query == "guitar" and r.context_query == "" and r.sound_only


def test_context_query_fallback_without_model_fields():
    r = qr.Rewrite(captions=[], hints={"sound_terms": ["whistling"]}, query="whistling car commercial 2008", raw={})
    assert r.sound_query == "whistling"
    assert r.context_query == "car commercial 2008"
    assert not r.sound_only


def test_cards_sound_and_context_text():
    from catalog.cards import card_text, context_text, sound_text
    desc = {"description": "bright acoustic guitar riff", "instruments": ["acoustic guitar", "drums"], "tempo": "mid", "vocals": "male",
            "mood": ["sunny"], "theme": "a road trip with friends"}
    text = card_text({"title": "Guitar Town", "artist": "X"}, None, desc)
    assert "Description audio" in text and "Thème des paroles" in text
    ctx = context_text(text)
    assert "Description audio" not in ctx and "Guitar Town" in ctx and "road trip" in ctx
    snd = sound_text(desc)
    assert "acoustic guitar" in snd and "Guitar Town" not in snd and "road trip" not in snd
    assert sound_text(None) is None
