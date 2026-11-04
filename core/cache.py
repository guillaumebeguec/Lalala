"""Disk cache keyed by content hash, for every expensive computation and API call.

Key = SHA-256 of the content (text, bytes or file) and of the parameters
(model, dimension, task...). Each namespace (`namespace`) has its own folder
in `data/cache/`. Values are serialised as JSON (dicts, lists, numbers) or
as raw bytes; numpy arrays go through `np.save`.

Usage:

    cache = DiskCache("gemini_text")
    key = cache.key("some text", model="gemini-embedding-2", dim=768)
    vec = cache.get_array(key)
    if vec is None:
        vec = compute()
        cache.put_array(key, vec)
"""

from __future__ import annotations

import hashlib
import json
import io
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_settings


def content_hash(content: str | bytes | Path, **params: Any) -> str:
    """SHA-256 of the content and the parameters (sorted to be deterministic).

    - `str`  : encoded as UTF-8;
    - `bytes`: as is;
    - `Path` : file contents (read in chunks).
    """
    h = hashlib.sha256()
    if isinstance(content, Path):
        with content.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    elif isinstance(content, bytes):
        h.update(content)
    else:
        h.update(str(content).encode("utf-8"))
    if params:
        h.update(b"\x00")
        h.update(json.dumps(params, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None):
        self.namespace = namespace
        self.root = (root or get_settings().cache_dir) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- keys ----
    def key(self, content: str | bytes | Path, **params: Any) -> str:
        return content_hash(content, namespace=self.namespace, **params)

    def _path(self, key: str, ext: str) -> Path:
        # Two levels of sub-folders to avoid huge directories.
        return self.root / key[:2] / f"{key}{ext}"

    # ---- JSON ----
    def get(self, key: str) -> Any | None:
        p = self._path(key, ".json")
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: Any) -> None:
        p = self._path(key, ".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    # ---- bytes ----
    def get_bytes(self, key: str, ext: str = ".bin") -> bytes | None:
        p = self._path(key, ext)
        return p.read_bytes() if p.exists() else None

    def put_bytes(self, key: str, data: bytes, ext: str = ".bin") -> Path:
        p = self._path(key, ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(ext + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p

    # ---- numpy ----
    def get_array(self, key: str) -> np.ndarray | None:
        p = self._path(key, ".npy")
        return np.load(p) if p.exists() else None

    def put_array(self, key: str, arr: np.ndarray) -> None:
        p = self._path(key, ".npy")
        p.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr))
        tmp = p.with_suffix(".npy.tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(p)

    def has(self, key: str, ext: str = ".json") -> bool:
        return self._path(key, ext).exists()

    def clear(self) -> int:
        n = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                p.unlink()
                n += 1
        return n
