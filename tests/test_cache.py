import numpy as np

from core.cache import DiskCache, content_hash


def test_hash_depends_on_params():
    a = content_hash("hello", model="m", dim=8)
    b = content_hash("hello", model="m", dim=16)
    c = content_hash("hello", dim=8, model="m")  # ordre des paramètres indifférent
    assert a != b and a == c


def test_hash_of_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert content_hash(p) == content_hash(b"abc")


def test_json_roundtrip():
    c = DiskCache("test_json")
    k = c.key("texte", task="query")
    assert c.get(k) is None
    c.put(k, {"a": [1, 2], "b": "é"})
    assert c.get(k) == {"a": [1, 2], "b": "é"}
    assert c.has(k)


def test_array_and_bytes_roundtrip():
    c = DiskCache("test_arr")
    k = c.key(b"audio-bytes", model="clap")
    v = np.random.rand(4).astype("float32")
    c.put_array(k, v)
    np.testing.assert_array_equal(c.get_array(k), v)
    c.put_bytes(k, b"\x00\x01", ext=".mp3")
    assert c.get_bytes(k, ext=".mp3") == b"\x00\x01"
    assert c.clear() == 2
