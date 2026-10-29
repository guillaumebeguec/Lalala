"""Calcule les embeddings du catalogue pour un embedder : audio des extraits (lus en
mémoire, jamais écrits sur disque) et texte des fiches.

    uv run python scripts/build_embeddings.py --embedder clap|peav|gemma|gemini --subset dev|full [--modality audio,text] [--limit N]

Reprise : chaque vecteur est en cache (data/cache/emb_<embedder>/, clé = hash du contenu).
Sortie : data/index/<embedder>_<modalité>/ (FAISS) via core.index.VectorIndex, et un
résumé temps/couverture dans data/index/<embedder>_<modalité>/stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from catalog.previews import fetch_audio, load_previews  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.embedders.base import get_embedder  # noqa: E402
from core.index import VectorIndex, index_dir  # noqa: E402

log = logging.getLogger("build_embeddings")


def audio_vectors(emb, cat: pd.DataFrame, prev: pd.DataFrame, batch: int, limit: int | None) -> tuple[list[str], np.ndarray, dict]:
    ok = prev[prev.status == "ok"].set_index("track_id")
    ids = [t for t in cat.track_id if t in ok.index]
    if limit:
        ids = ids[:limit]
    vecs, done_ids, n_fail = [], [], 0
    t0 = time.time()
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        items, keep = [], []
        for tid in chunk:
            try:
                data = fetch_audio(tid, ok.at[tid, "preview_url"])
            except Exception as e:  # noqa: BLE001
                log.warning("extrait %s : %s", tid, e)
                data = None
            if data:
                items.append(data)
                keep.append(tid)
            else:
                n_fail += 1
        if items:
            v = emb.embed_audio(items, batch_size=batch)
            vecs.append(v)
            done_ids += keep
        if (start // batch) % 20 == 0:
            el = time.time() - t0
            log.info("audio %d / %d (%.2f s/extrait, échecs %d)", len(done_ids), len(ids), el / max(len(done_ids), 1), n_fail)
    V = np.concatenate(vecs) if vecs else np.zeros((0, emb.dim), np.float32)
    return done_ids, V, {"n": len(done_ids), "n_fail": n_fail, "seconds": round(time.time() - t0, 1)}


def text_vectors(emb, cards: pd.DataFrame, batch: int, limit: int | None) -> tuple[list[str], np.ndarray, dict]:
    df = cards.head(limit) if limit else cards
    t0 = time.time()
    V = emb.embed_text(df.card_text.tolist(), task="document", batch_size=batch)
    return df.track_id.tolist(), V, {"n": len(df), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", required=True, choices=["clap", "peav", "gemma", "gemini"])
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--modality", default="audio,text")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    emb = get_embedder(args.embedder)
    cat = cb.load(args.subset)
    mods = {m.strip() for m in args.modality.split(",")}
    if "audio" in mods:
        prev = load_previews()
        ids, V, stats = audio_vectors(emb, cat, prev, args.batch, args.limit)
        VectorIndex.build(args.embedder, "audio", ids, V).save()
        stats.update(subset=args.subset, dim=int(V.shape[1]) if len(V) else emb.dim)
        (index_dir(args.embedder, "audio") / "stats.json").write_text(json.dumps(stats, indent=1))
        log.info("audio : %s", stats)
    if "text" in mods:
        cards_path = get_settings().data_dir / "catalog" / f"cards_{args.subset}.parquet"
        if not cards_path.exists():
            log.error("fiches absentes : %s (scripts/build_cards.py)", cards_path)
            return 1
        cards = pd.read_parquet(cards_path)
        cards = cards[cards.track_id.isin(set(cat.track_id))]
        ids, V, stats = text_vectors(emb, cards, max(args.batch, 16), args.limit)
        VectorIndex.build(args.embedder, "text", ids, V).save()
        stats.update(subset=args.subset, dim=int(V.shape[1]) if len(V) else emb.dim)
        (index_dir(args.embedder, "text") / "stats.json").write_text(json.dumps(stats, indent=1))
        log.info("texte : %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
