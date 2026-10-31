import numpy as np

from core.fusion_learned import LearnedFusion


def test_learned_fusion_learns_to_trust_a_good_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from core import config
    config.get_settings.cache_clear()
    rng = np.random.default_rng(0)
    lf = LearnedFusion(branches=["good", "noise"])
    meta = {f"t{i}": {"year": 1980 + i, "language": "english"} for i in range(30)}
    samples = []
    for q in range(60):
        ans = f"t{rng.integers(0, 30)}"
        others = [f"t{i}" for i in rng.permutation(30) if f"t{i}" != ans][:9]
        good = [(ans, 0.9)] + [(o, 0.5 - 0.03 * k) for k, o in enumerate(others)] if rng.random() < 0.8 else [(o, 0.5) for o in others] + [(ans, 0.1)]
        noise = [(f"t{i}", 0.5) for i in rng.permutation(30)[:10]]
        samples.append(({"good": good, "noise": noise}, {"decade": "1980s"}, meta, "text", {ans}))
    info = lf.fit(samples)
    assert info["coef"]["good:inv_rank"] > info["coef"]["noise:inv_rank"]
    ranked = lf.rank({"good": [("t3", 0.9), ("t4", 0.4)], "noise": [("t4", 0.5), ("t9", 0.5)]}, {}, meta)
    assert ranked[0][0] == "t3"
    lf.save()
    assert LearnedFusion.load().branches == ["good", "noise"]
