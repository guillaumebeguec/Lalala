import pandas as pd

from catalog.build import Track, dedup, make_dev_subset, to_frame


def test_dedup_keeps_priority_and_merges():
    a = Track(title="Hello (Remastered)", artist="Adele", deezer_id=1, rank=100, sources=["hits_intl"], chart_years=[2015])
    b = Track(title="Hello", artist="Adele", wasabi_id="w1", rank=50, sources=["tot"], year=2015, genre="Pop")
    c = Track(title="Hello", artist="The Adele", deezer_id=3, rank=10, sources=["distract"])
    out = dedup([a, b, c])
    assert len(out) == 1
    t = next(iter(out.values()))
    assert t.sources[0] == "tot" and set(t.sources) == {"tot", "hits_intl", "distract"}
    assert t.deezer_id == 1 and t.wasabi_id == "w1" and t.rank == 100 and t.chart_years == [2015]
    assert t.track_id == "dz1"


def test_to_frame_year_fallback_and_dev_subset():
    tracks = [Track(title=f"t{i}", artist=f"a{i}", deezer_id=i + 1, sources=["tot" if i < 10 else "hits_intl"],
                    chart_years=[1990] if i >= 10 else [], year=2000 if i < 10 else None) for i in range(100)]
    df = to_frame(tracks)
    assert df.year_source.tolist()[:10] == ["meta"] * 10 and df.year.iloc[20] == 1990
    mask = make_dev_subset(df, size=30, n_tot=5)
    sub = df[mask]
    assert len(sub) == 30 and sub.is_tot_answer.sum() == 5
    # déterministe
    assert (make_dev_subset(df, size=30, n_tot=5) == mask).all()
