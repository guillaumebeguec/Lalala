import json

import pandas as pd

from catalog import wasabi


SONG = {
    "_id": {"$oid": "abc"}, "title": "Turn It Up", "name": "A", "albumTitle": "How Ace", "id_album": {"$oid": "alb"},
    "publicationDate": "1998-06-22", "publicationDateAlbum": "1997", "album_genre": "Alternative Rock",
    "language": "", "language_detect": "english", "id_song_deezer": "67354194", "isrc": "GBAAP9700050",
    "id_song_musicbrainz": "3db6", "rank": 261631, "bpm": "77", "length": "93", "preview": "http://x/1.mp3",
    "availableCountries": ["FR", "DE"], "urlWikipedia": "", "lyrics": "SECRET LYRICS", "summary": ["lyric line"],
    "valence": "", "valence_predicted": 0.4, "arousal": 0.7, "explicitLyrics": False, "has_social_tags": True,
}


def test_project_song_fields_and_no_lyrics():
    r = wasabi.project_song(SONG)
    assert r["wasabi_id"] == "abc" and r["artist"] == "A" and r["year"] == 1997 and r["deezer_year"] == 1998
    assert r["deezer_id"] == 67354194 and r["available_fr"] is True and r["n_countries"] == 2
    assert r["language"] == "english" and r["valence"] == 0.4 and r["url_wikipedia"] is None
    assert "lyrics" not in r and "summary" not in r
    assert set(r) == set(wasabi.SONG_SCHEMA.names)


def test_year_fallback_to_deezer_date():
    r = wasabi.project_song({**SONG, "publicationDateAlbum": ""})
    assert r["year"] == 1998
    assert wasabi.project_song({**SONG, "publicationDate": "", "publicationDateAlbum": ""})["year"] is None


def test_abstract_from_rdf():
    rdf = "<x><dbo:abstract xml:lang='en'>Hello &amp; world.  Twice.</dbo:abstract></x>"
    assert wasabi._abstract(rdf) == "Hello & world. Twice."
    assert wasabi._abstract("") is None and wasabi._abstract("<x/>") is None


def test_extract_roundtrip(tmp_path):
    src = tmp_path / "json"
    src.mkdir()
    (src / "song.json").write_text(json.dumps([SONG, {**SONG, "_id": {"$oid": "def"}, "title": "Foghorn"}]))
    (src / "album.json").write_text(json.dumps([{"_id": {"$oid": "alb"}, "title": "How Ace", "name": "A",
                                                 "id_artist": {"$oid": "art"}, "genre": "Rock", "publicationDate": "1997"}]))
    (src / "artist-without-members.json").write_text(json.dumps([{"_id": {"$oid": "art"}, "name": "A", "genres": ["Rock"],
                                                                  "locationInfo": ["England", "Leeds"]}]))
    (src / "artist-members.json").write_text("[]")
    dst = tmp_path / "out"
    counts = wasabi.extract(src, dst)
    assert counts == {"songs": 2, "albums": 1, "artists": 1}
    s = pd.read_parquet(dst / "songs.parquet")
    assert list(s.title) == ["Turn It Up", "Foghorn"] and "lyrics" not in s.columns
    a = pd.read_parquet(dst / "artists.parquet")
    assert a.location[0] == "England, Leeds" and list(a.genres[0]) == ["Rock"]
    # reprise : rien n'est recalculé
    assert wasabi.extract(src, dst) == counts
