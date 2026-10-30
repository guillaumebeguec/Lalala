from testsets.chad import parse_artist_title


def test_parse_youtube_titles():
    assert parse_artist_title("Kim Carnes - Bette Davis Eyes (Official Video)") == ("Kim Carnes", "Bette Davis Eyes")
    assert parse_artist_title("Daft Punk – Get Lucky [Official Audio]") == ("Daft Punk", "Get Lucky")
    assert parse_artist_title("Adele - Hello") == ("Adele", "Hello")
    assert parse_artist_title("Some random vlog") is None
