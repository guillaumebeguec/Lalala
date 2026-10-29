import pytest

from core.fusion import normalize_scores, rrf, soft_bonus


def test_rrf_basic_and_ties():
    a = [("x", 0.9), ("y", 0.8), ("z", 0.1)]
    b = [("y", 5.0), ("w", 4.0)]
    out = rrf([a, b], k=60)
    ids = [i for i, _ in out]
    assert ids[0] == "y"  # présent dans les deux listes
    assert set(ids) == {"x", "y", "z", "w"}
    assert abs(dict(out)["y"] - (1 / 62 + 1 / 61)) < 1e-12
    assert rrf([], k=60) == []
    assert rrf([[("a", 1), ("a", 0.5)]], k=10) == [("a", 1 / 11)]  # doublon ignoré


def test_rrf_weights_and_top():
    a = [("x", 1.0)]
    b = [("y", 1.0)]
    assert rrf([a, b], k=1, weights=[1.0, 3.0])[0][0] == "y"
    assert len(rrf([a, b], k=1, top=1)) == 1
    with pytest.raises(ValueError):
        rrf([a, b], weights=[1.0])


def test_normalize_scores():
    assert normalize_scores([("a", 2.0), ("b", 1.0)]) == [("a", 1.0), ("b", 0.0)]
    assert normalize_scores([("a", 1.0), ("b", 1.0)]) == [("a", 1.0), ("b", 1.0)]


def test_soft_bonus_never_excludes():
    meta = {"a": {"year": 1985, "language": "french"}, "b": {"year": 2001, "language": "english"}, "c": {}}
    hits = [("b", 1.0), ("a", 0.9), ("c", 0.8)]
    out = soft_bonus(hits, meta, {"decade": "1980s", "language": "french"})
    assert [i for i, _ in out][0] == "a" and len(out) == 3
    assert dict(out)["c"] == 0.8  # sans métadonnées : inchangé
    assert soft_bonus(hits, meta, {}) == hits
