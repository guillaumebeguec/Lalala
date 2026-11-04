import pytest

from testsets import tot_music


def _sample_row():
    return {
        "id": "abc", "wasabi_2.0_id": "5714", "wasabi_isrc": "USX1", "query_types": ["descriptive"], "split": "test",
        "title": "Need help", "text": "Male singer, 80s rock, lyric about a back door.",
        "raw_title": "[TOMT][SONG][ROCK][80s (maybe)] Need help",
        "spotify_resolved": {"title": "Rag Doll", "items": [{"url": "u", "isrc": ["USX1"],
                             "artists": [{"name": "Aerosmith", "uri": "x"}]}]},
    }


def test_parse_row_and_tags():
    q = tot_music._parse_row(_sample_row())
    assert q.answer_title == "Rag Doll" and q.answer_artists == ["Aerosmith"]
    assert q.tag_decade == "1980s" and q.tag_genre == "ROCK"
    assert "Rag Doll" not in q.visible_text and "Aerosmith" not in q.visible_text


@pytest.mark.parametrize("title,decade", [
    ("[TOMT][SONG][2000s] x", "2000s"), ("[TOMT] [SONG] [90's] x", "1990s"),
    ("[TOMT][SONG][1970s] x", "1970s"), ("[TOMT][SONG] x", None),
])
def test_decade_tags(title, decade):
    assert tot_music._parse_tags(title)[0] == decade


def test_load_real_file_if_present():
    import os
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "raw" / "tot" / "data.jsonl"
    if not p.exists():
        pytest.skip("ToT_Music not downloaded")
    os.environ["DATA_DIR"] = str(p.parent.parent.parent)
    from core import config
    config.get_settings.cache_clear()
    qs = tot_music.load(download_if_missing=False)
    assert len(qs) > 2000
    assert all(q.answer_wasabi_id for q in qs)
