"""Empreintes de paroles non réversibles (hachés de 4-grammes de mots) pour les morceaux du catalogue.

    uv run python scripts/build_lyrics_hashes.py [--n 4]

Lit en streaming `data/raw/wasabi/json/song.json` (champ `lyrics`), ne garde pour chaque morceau
du catalogue (par `wasabi_id`) que l'ensemble des hachés 64 bits de ses n-grammes de mots
normalisés. **Aucun texte de paroles n'est écrit** : les hachés sont à sens unique et ne
permettent pas de reconstituer les paroles (CLAUDE.md : pas de paroles stockées). Sert à relier un
fragment chanté transcrit par Whisper à un morceau. Sortie : data/index/lyrics_ngrams.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog import wasabi  # noqa: E402
from core.config import get_settings  # noqa: E402

log = logging.getLogger("lyrics_hashes")
_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    t = html.unescape(_TAG_RE.sub(" ", text or ""))
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return [w.strip("'") for w in _WORD_RE.findall(t) if w.strip("'")]


def ngram_hashes(words: list[str], n: int = 4) -> np.ndarray:
    if len(words) < n:
        return np.zeros(0, dtype=np.uint64)
    out = set()
    for i in range(len(words) - n + 1):
        h = hashlib.blake2b(" ".join(words[i : i + n]).encode("utf-8"), digest_size=8).digest()
        out.add(int.from_bytes(h, "little"))
    return np.array(sorted(out), dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cat = cb.load("full")
    by_wid = {w: t for w, t in zip(cat.wasabi_id, cat.track_id) if isinstance(w, str) and w}
    log.info("catalogue : %d morceaux, %d avec identifiant WASABI", len(cat), len(by_wid))
    src = wasabi.raw_dir() / "json" / "song.json"
    t0 = time.time()
    ids, hashes, n_words = [], [], []
    seen = 0
    for d in wasabi._stream(src):
        seen += 1
        wid = wasabi._oid(d.get("_id"))
        if wid not in by_wid:
            continue
        words = normalize_words(d.get("lyrics") or "")
        if len(words) < args.n:
            continue
        ids.append(by_wid[wid]); hashes.append(ngram_hashes(words, args.n)); n_words.append(len(words))
        if len(ids) % 2000 == 0:
            log.info("%d morceaux avec empreinte (%d lus, %.0f s)", len(ids), seen, time.time() - t0)
    out = get_settings().index_dir / "lyrics_ngrams.parquet"
    table = pa.table({"track_id": ids, "n_words": n_words, "hashes": pa.array([h.tolist() for h in hashes], type=pa.list_(pa.uint64()))})
    pq.write_table(table, out, compression="zstd")
    log.info("écrit %s : %d morceaux, %d n-grammes au total, %.0f s", out, len(ids), sum(len(h) for h in hashes), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
