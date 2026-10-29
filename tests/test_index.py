import numpy as np

from core.index import BM25Index, VectorIndex, tokenize


def test_vector_index_roundtrip():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 8)).astype(np.float32)
    ids = [f"t{i}" for i in range(50)]
    vi = VectorIndex.build("dummy", "audio", ids, vecs)
    hits = vi.search(vecs[7], k=3)[0]
    assert hits[0][0] == "t7" and abs(hits[0][1] - 1.0) < 1e-5
    vi.save()
    vi2 = VectorIndex.load("dummy", "audio")
    assert vi2.ids == ids and vi2.search(vecs[3], k=1)[0][0][0] == "t3"
    assert VectorIndex.exists("dummy", "audio") and not VectorIndex.exists("dummy", "text")
    assert np.allclose(vi2.vector_of("t3"), vecs[3] / np.linalg.norm(vecs[3]), atol=1e-5)


def test_bm25_and_tokenize():
    assert tokenize("Les Démons de Minuit, 1986 !") == ["demons", "minuit", "1986"]
    idx = BM25Index(["a", "b", "c"], ["Titre : Bette Davis Eyes Artiste : Kim Carnes Année : 1981",
                                     "Titre : Les Démons de Minuit Artiste : Images Année : 1986",
                                     "Titre : Hello Artiste : Adele Année : 2015"])
    assert idx.search("démons minuit")[0][0] == "b"
    assert idx.search("kim carnes 1981")[0][0] == "a"
    assert idx.search("zzzz") == []
    p = idx.save()
    assert BM25Index.load(p).search("adele")[0][0] == "c"
