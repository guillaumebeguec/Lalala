import math

from finder.belief import DONT_KNOW, OTHER, Belief, REPERTOIRE, entropy, ig_policy


META = {
    "a": {"year": 1984, "language": "english", "genre": "Pop", "vocals": "female"},
    "b": {"year": 1986, "language": "english", "genre": "Rock", "vocals": "male"},
    "c": {"year": 1999, "language": "french", "genre": "Pop", "vocals": "male"},
    "d": {"year": 2005, "language": "french", "genre": "Hip Hop", "vocals": "male"},
}
SCORED = [("a", 0.9), ("b", 0.85), ("c", 0.8), ("d", 0.75)]


def test_prior_and_entropy():
    b = Belief.from_candidates(SCORED, META, temperature=1.0, p_other=0.2)
    assert abs(sum(b.probs) - 1) < 1e-9 and b.ids[-1] == OTHER and abs(b.p_other - 0.2) < 1e-9
    assert entropy(b.probs) > 2.0


def test_best_question_splits_candidates_and_update_lowers_entropy():
    b = Belief.from_candidates(SCORED, META, temperature=1.0, p_other=0.1)
    q = b.best_question()
    assert q is not None and q["expected_gain"] > 0
    assert q["key"] in ("decade", "language", "genre", "vocals")
    h0 = entropy(b.probs)
    rec = b.update("language", "français", ["français", "anglais"])
    assert rec["entropy_after"] < h0
    top, p = b.top
    assert top in ("c", "d") and b.probs[b.ids.index("a")] < 0.1
    assert "language" in b.asked and b.best_question()["key"] != "language"


def test_dont_know_is_uninformative():
    b = Belief.from_candidates(SCORED, META, temperature=1.0)
    before = list(b.probs)
    b.update("decade", DONT_KNOW, ["1980s", "1990s", "2000s"])
    assert all(abs(x - y) < 1e-9 for x, y in zip(before, b.probs))


def test_other_hypothesis_grows_on_contradiction():
    b = Belief.from_candidates(SCORED, META, temperature=1.0, p_other=0.1, epsilon=0.05)
    b.update("language", "autre langue", ["français", "anglais", "autre langue"])
    assert b.p_other > 0.5  # no candidate fits: "other" dominates


def test_merge_candidates_redistributes_other_mass():
    b = Belief.from_candidates(SCORED[:2], META, temperature=1.0, p_other=0.4)
    b.merge_candidates([("c", 0.9), ("d", 0.1)], META, temperature=1.0)
    assert b.ids[-1] == OTHER and abs(sum(b.probs) - 1) < 1e-9 and abs(b.p_other - 0.2) < 1e-9
    assert b.probs[b.ids.index("c")] > b.probs[b.ids.index("d")]


def test_policy_stops_when_confident():
    class C:
        def __init__(self, i, s, y): self.track_id, self.score, self.year = i, s, y
    class T:
        def meta_for_belief(self, i): return META[i]
    class S:
        candidates = {"a": C("a", 5.0, 1984), "b": C("b", 0.1, 1986)}
        questions = []
    class A:
        tools = T()
    pol = ig_policy(stop_prob=0.7, temperature=0.1, p_other=0.05)
    assert pol(A(), S()) is None  # "a" already dominates
    S2 = type("S2", (), {"candidates": {k: C(k, 1.0, META[k]["year"]) for k in META}, "questions": []})()
    q = pol(A(), S2)
    assert q and q["options"] and "expected_gain" in q
