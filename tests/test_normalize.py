from catalog.normalize import dedup_key, fold, norm_artist, norm_title, similarity, is_version_of


def test_fold():
    assert fold("Élégie – Café & Thé!") == "elegie cafe and the"


def test_norm_title_strips_versions():
    assert norm_title("Ashes to Ashes (2017 Remaster)") == "ashes to ashes"
    assert norm_title("Another Brick In The Wall, Pt. 2 (2011 Remastered Version)") == "another brick in the wall pt 2"
    assert norm_title("One More Time - Short Radio Edit") == "one more time"
    assert norm_title("Get Lucky (feat. Pharrell Williams)") == "get lucky"
    assert norm_title("(I Can't Get No) Satisfaction") == "satisfaction"


def test_norm_artist():
    assert norm_artist("The Police") == "police"
    assert norm_artist("Daft Punk feat. Pharrell") == "daft punk"
    assert dedup_key("Hello", "Adele") == "hello|adele"


def test_similarity_and_version():
    assert similarity("Bette Davis Eyes", "Bette Davis Eyes (Remastered)") > 0.7
    assert similarity("abc", "") == 0.0
    assert is_version_of("Imagine (Live at Madison Square Garden)") and not is_version_of("Imagine")
