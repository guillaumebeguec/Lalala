"""Test anti-fuite : la réponse attendue n'apparaît jamais dans le prompt de l'utilisateur simulé."""
import pytest

from finder.simulated_user import DONT_KNOW, SimulatedUser, _snap, from_tot
from testsets import tot_music


def _tot_query():
    return tot_music._parse_row({
        "id": "x", "wasabi_2.0_id": "5714dec325ac0d8aee386d8c", "wasabi_isrc": "USGF18716203", "query_types": ["descriptive"], "split": None,
        "title": "Song with a whistle", "text": "80s song, male voice, whistling intro, used in a car ad.",
        "raw_title": "[TOMT][SONG][80s] Song with a whistle",
        "spotify_resolved": {"title": "Secret Title", "items": [{"url": "u", "isrc": ["USGF18716203"],
                             "artists": [{"name": "Hidden Artist", "uri": "x"}]}]},
    })


@pytest.mark.parametrize("informed", [False, True])
def test_answer_never_in_prompt(informed):
    q = _tot_query()
    u = from_tot(q, informed=informed, year=1985, language="english")
    u.history.append({"q": "Décennie ?", "a": "1980s"})
    msgs = u.prompt_messages("Voix ?", ["masculine", "féminine", "instrumental"])
    blob = "\n".join(m["content"] for m in msgs).lower()
    for secret in (q.answer_title, q.answer_artists[0], q.answer_isrc, q.answer_wasabi_id):
        assert secret.lower() not in blob
    assert q.text.lower() in blob
    assert (("1985" in blob) == informed) and (("english" in blob) == informed)


def test_uninformed_drops_known_fields():
    u = SimulatedUser(description="d", informed=False, known_year=1999, known_language="french")
    assert u.known_year is None and u.known_language is None


def test_snap():
    assert _snap("1980s", ["1970s", "1980s"]) == "1980s"
    assert _snap("Je ne sais pas.", ["a", "b"]) == DONT_KNOW
    assert _snap("plutôt masculine", ["masculine", "féminine"]) == "masculine"
    assert _snap("banane", ["a", "b"]) == DONT_KNOW


@pytest.mark.llm
def test_simulated_user_answers_from_text_only():
    u = SimulatedUser(description="Une chanson des années 80 avec un sifflement au début, chantée par un homme.")
    assert u.answer("Dans quelle décennie est sortie la chanson ?", ["1970s", "1980s", "1990s"]) == "1980s"
    assert u.answer("La voix est-elle masculine ou féminine ?", ["masculine", "féminine", "instrumental"]) == "masculine"
    assert u.answer("La chanson est-elle en français ou en anglais ?", ["français", "anglais"]) == DONT_KNOW
