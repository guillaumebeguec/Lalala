import numpy as np

from core.embedders.base import Embedder, l2_normalize


class Dummy(Embedder):
    name = "dummy"
    dim = 4

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _embed_audio(self, items):
        self.calls += 1
        return np.array([[len(b), 1, 0, 0] for b in items], dtype=np.float32)

    def _embed_text(self, texts, task):
        self.calls += 1
        return np.array([[len(t), 0, 1, 1 if task == "query" else 0] for t in texts], dtype=np.float32)


def test_l2_normalize():
    v = l2_normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0) and v.dtype == np.float32
    assert l2_normalize(np.zeros(3)).shape == (1, 3)


def test_cache_and_batching():
    e = Dummy()
    a = e.embed_audio([b"abc", b"defgh", b"ij"], batch_size=2)
    assert a.shape == (3, 4) and e.calls == 2
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    e.calls = 0
    b = e.embed_audio([b"ij", b"abc"])
    assert e.calls == 0 and np.allclose(b[0], a[2]) and np.allclose(b[1], a[0])
    q = e.embed_text(["hello"], task="query")
    d = e.embed_text(["hello"], task="document")
    assert not np.allclose(q, d) and e.calls == 2
    e.embed_text(["hello"], task="query")
    assert e.calls == 2


def test_signal_items_are_cached_by_content():
    e = Dummy()
    x = (np.ones(100, dtype=np.float32), 16000)
    e._embed_audio = lambda items: np.ones((len(items), 4), dtype=np.float32)
    e.embed_audio([x])
    k1 = e._audio_key(x)
    assert e._cache.get_array(k1) is not None
    assert e._audio_key((np.ones(100, dtype=np.float32), 8000)) != k1
