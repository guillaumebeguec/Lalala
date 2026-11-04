"""Utilities shared by the test sets: frozen JSONL output, seed, statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.config import get_settings

SEED = 20261028


def testsets_dir() -> Path:
    d = get_settings().testsets_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
